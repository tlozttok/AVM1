"""$MEM.model_catalog：可用的 LLM 模型与本地 PLM 清单。

这份目录是给对话看的——它让 para 在写参数时能自己查"这家接受哪些参数""本地有
哪些 PLM"，而不是靠试错。两侧的说明各写各的：LLM 侧是上游 API 参数，PLM 侧是
本地实现。

两类内容的来源不同：
- 静态部分（PLM 注册表、参数清单）不依赖网络，装载后装一次就够；
- 上游清单要拉 `/models`，取不到就把 source 标成 unavailable——**绝不拦启动**，
  也不拦模型名（模型名校验在 Core._check_model_namespace）。

这里只读写内存里的目录节点与打印提示，判权/记日志都不归它管。
"""

import json
import sys

from .python_server import HOST_ONLY_PLMS, PLM_REGISTRY
from .types import MetaDict, MetaList

MODEL_CATALOG_KEY = "model_catalog"

_CATALOG_META = "模型目录：可用的 LLM 模型与本地 PLM；两侧都写明各自接受的参数"
_PLM_PARAMS_NOTE = "只有 model 有效；tools 由内核注入；LLM 参数无效"
_PLM_OPERATOR_NOTE = "只能由宿主创建（plm.operator 需要注入操作员通道）；不能由镜像声明"


def _first_line(text: str, limit: int = 72) -> str:
    """取文档串首行，超长截断——目录里的说明要能一眼看完。"""
    line = (text or "").strip().split("\n", 1)[0].strip()
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _raw_response_json(raw):
    """从 SDK 的"原始响应"对象里取回 JSON 对象，兼容 openai 各版本。

    - openai 2.x 的 `LegacyAPIResponse`：没有 `.json()`，有 `.http_response`（httpx 响应）与 `.text`；
    - openai 1.x：有 `.json()`；
    - 未来主版本把 `.text`/`.content` 改成方法——两者都按可调用或属性处理。
    """
    http = getattr(raw, "http_response", None)
    json_fn = getattr(http, "json", None) if http is not None else None
    if callable(json_fn):
        return json_fn()
    json_fn = getattr(raw, "json", None)
    if callable(json_fn):
        return json_fn()
    text = getattr(raw, "text", None)
    if callable(text):
        text = text()
    if isinstance(text, str) and text:
        return json.loads(text)
    content = getattr(raw, "content", None)
    if callable(content):
        content = content()
    if isinstance(content, (bytes, bytearray)):
        return json.loads(bytes(content).decode("utf-8"))
    raise TypeError(f"无法从 {type(raw).__name__} 取原始 JSON 响应体")


class ModelCatalog:
    """模型目录的安装、刷新与"默认 para 用了清单外模型"的提醒。

    参数清单由构造方注入（LMU 认得哪些 API 参数），不在本模块再抄一份：
    抄一份就会与真正做过滤的 LMU 漂移。
    """

    def __init__(self, core, api_param_keys, numeric_params):
        self.core = core
        self.api_param_keys = set(api_param_keys)
        self.numeric_params = set(numeric_params)

    def install(self) -> None:
        """安装/补齐模型目录的静态部分。幂等；镜像或检查点里已有的字段不覆盖。

        整树替换（load_image）之后必须重装一次，否则镜像会把启动节点冲掉。
        静态部分不依赖网络：PLM 注册表、para 能用的 API 参数、必填字段。
        """
        core = self.core
        root = core.mem.root
        catalog = root.get(MODEL_CATALOG_KEY)
        if not isinstance(catalog, MetaDict):
            catalog = MetaDict(data={}, metadata=_CATALOG_META)
            root[MODEL_CATALOG_KEY] = catalog
        llm = catalog.get("llm")
        if not isinstance(llm, MetaDict):
            llm = MetaDict(data={}, metadata="可用的 LLM 模型；source 说明这份清单的来源")
            catalog["llm"] = llm
        llm.setdefault("source", "unavailable")
        llm.setdefault("required", "model")
        params = llm.get("params")
        if not isinstance(params, MetaDict):
            params = MetaDict(data={}, metadata="可以写进 para 的 API 参数（model 之外全部可选）")
            llm["params"] = params
        for name in sorted(self.api_param_keys):
            if name not in params:
                params[name] = "数值型" if name in self.numeric_params else "字符串/原样"
        if not isinstance(llm.get("models"), MetaList):
            llm["models"] = MetaList(data=[], metadata="上游模型清单；每项一个 id")
        if not isinstance(catalog.get("plm"), MetaList):
            catalog["plm"] = MetaList(
                data=[
                    self._plm_catalog_entry(name, cls)
                    for name, cls in PLM_REGISTRY.items()
                    if name not in HOST_ONLY_PLMS  # 宿主独占的 PLM 不列给对话
                ] + [
                    # 镜像声明的 PLM 也列进来（同名内建优先，装载期已拒绝重名）
                    self._plm_catalog_entry(name, cls)
                    for name, cls in core.plms.items()
                    if name not in PLM_REGISTRY
                ],
                metadata="PLM 注册表：本地实现，不需要网络",
            )

    @staticmethod
    def _plm_catalog_entry(name: str, cls) -> MetaDict:
        entry = MetaDict(
            data={"name": name, "required": "model"},
            metadata=_first_line(getattr(cls, "__doc__", "")) or name,
        )
        entry["params"] = _PLM_OPERATOR_NOTE if name == "plm.operator" else _PLM_PARAMS_NOTE
        return entry

    def refresh(self, timeout: float = 10.0) -> dict:
        """拉一次上游 `/models` 写进目录。任何失败都收敛成 source="unavailable"，不抛。

        防御按讨论案第五节：自己校验信封、除 id 外全部按可选字段读、忽略未知字段。
        清单只当参考——取不到不拦启动，也不拦模型名（后者见 _check_model_namespace）。
        """
        core = self.core
        self.install()
        llm = core.mem[MODEL_CATALOG_KEY]["llm"]

        def _unavailable(reason: str) -> dict:
            llm["source"] = "unavailable"
            llm["reason"] = reason
            print(
                f"[AVM] 模型目录：没能取到上游清单（{reason}），本轮不校验 LLM 模型名",
                file=sys.stderr,
            )
            return {"source": "unavailable", "reason": reason}

        try:
            # 用原始响应自己解析：SDK 默认模式对非 OpenAI 信封抛的是 AttributeError/TypeError，
            # 分不清"这家不是 OpenAI 形状"与"SDK 有 bug"；自己校验信封才有确定的诊断。
            raw = core.lmu.client.models.with_raw_response.list(timeout=timeout)
            payload = _raw_response_json(raw)
        except Exception as e:
            return _unavailable(f"{type(e).__name__}: {e}")
        if (
            not isinstance(payload, dict)
            or payload.get("object") != "list"
            or not isinstance(payload.get("data"), list)
        ):
            return _unavailable("上游 /models 的信封不是 OpenAI 形状（object=list 且 data 是数组）")
        entries = []
        skipped = 0
        for item in payload["data"]:
            if not isinstance(item, dict):
                skipped += 1
                continue
            model_id = item.get("id")
            if not isinstance(model_id, str) or not model_id:
                skipped += 1
                continue
            entry = MetaDict(data={"id": model_id}, metadata="上游模型 id")
            for key in ("name", "owned_by", "context_window", "max_output_tokens",
                        "input_modalities", "output_modalities"):
                value = item.get(key)
                if value is None:
                    continue
                # 内存约定：para 与目录里的值都以 str 存储
                entry[key] = ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)
            effort = item.get("effort")
            if isinstance(effort, dict):
                levels = effort.get("supported_levels")
                if isinstance(levels, list):
                    entry["reasoning_effort_levels"] = ", ".join(str(v) for v in levels)
                default_level = effort.get("default_level")
                if isinstance(default_level, str) and default_level:
                    entry["default_effort_level"] = default_level
            entries.append(entry)
        llm["models"] = MetaList(data=entries, metadata="上游模型清单；每项一个 id")
        llm["source"] = "api"
        if "reason" in llm:
            del llm["reason"]
        print(
            f"[AVM] 模型目录：从上游取到 {len(entries)} 个模型"
            + (f"（跳过 {skipped} 条形状不合法的条目）" if skipped else ""),
            file=sys.stderr,
        )
        self._warn_unlisted_para_model(llm)
        return {"source": "api", "count": len(entries), "skipped": skipped}

    def _warn_unlisted_para_model(self, llm: MetaDict) -> None:
        """默认 para 的 model 不在清单里时提醒运行者（例如用着已退役的别名）。"""
        core = self.core
        try:
            para = core.unwrap(core.para_ref, for_llm=False)
        except Exception:
            return
        model = para.get("model") if isinstance(para, MetaDict) else None
        if not isinstance(model, str) or not model:
            return
        listed = {e.get("id") for e in (llm.get("models") or []) if isinstance(e, MetaDict)}
        if model not in listed:
            print(
                f"[AVM] 模型目录：para {core.para_ref} 的 model {model!r} 不在上游清单里"
                f"（可能是拼写错误或已退役的别名）",
                file=sys.stderr,
            )
