"""main.py 调试入口测试（不调真实 API）"""

import json

from main import main


def _write(tmp_path, image):
    path = tmp_path / "image.json"
    path.write_text(json.dumps(image, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _no_init_image():
    return {
        "meta": {"name": "x", "version": 1},
        "mem": {
            "model_params": {
                "kind": "dict",
                "ctrl": {"type": "para"},
                "value": {"model": {"kind": "str", "value": "m"}},
            }
        },
        "para_ref": "$MEM.model_params",
    }


def test_main_frames_flag(tmp_path, capsys):
    rc = main([_write(tmp_path, _no_init_image()), "--frames"])
    assert rc == 0
    assert "监测帧" in capsys.readouterr().err


def test_main_persist_to(tmp_path, capsys):
    image = _no_init_image()
    image["persist_to"] = "out/mem.json"
    path = _write(tmp_path, image)
    rc = main([path])
    assert rc == 0
    assert (tmp_path / "out" / "mem.json").is_file()
    assert "内存已写回" in capsys.readouterr().err


def test_main_bad_image(tmp_path, capsys):
    path = _write(tmp_path, {"meta": {"version": 1}})  # 缺 mem 段
    rc = main([path])
    assert rc == 1
    assert "镜像错误" in capsys.readouterr().err


def test_main_loads_debug_config(tmp_path, capsys):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "image.json").write_text(json.dumps(_no_init_image()), encoding="utf-8")
    cfg_path = tmp_path / "debug.json"
    cfg_path.write_text(json.dumps({"image": "sub/image.json", "frames": True}), encoding="utf-8")

    rc = main(["--debug-config", str(cfg_path)])
    assert rc == 0
    assert "监测帧" in capsys.readouterr().err


def test_main_debug_config_missing_image(tmp_path, capsys):
    cfg_path = tmp_path / "debug.json"
    cfg_path.write_text(json.dumps({"frames": True}), encoding="utf-8")
    rc = main(["--debug-config", str(cfg_path)])
    assert rc == 1
    assert "缺少 image 字段" in capsys.readouterr().err


def _make_checkpoint(tmp_path):
    """造一个检查点文件：对话历史以完整 assistant 消息结束"""
    from avm.core import Core
    from avm.types import MetaDict

    core = Core()
    core.mem["model_params"] = MetaDict(data={"model": "m"}, ctrl={"type": "para"})
    conv = core.start("sys", "hello")
    conv.append_assistant_message("done")
    cp = tmp_path / "cp.json"
    core.save_conversation(conv.cid, str(cp))
    return str(cp)


def test_main_checkpoint_in_debug_config(tmp_path, capsys, monkeypatch):
    path = _write(tmp_path, _no_init_image())
    cp = _make_checkpoint(tmp_path)
    cfg_path = tmp_path / "debug.json"
    cfg_path.write_text(json.dumps({"image": "image.json", "checkpoint_in": "cp.json", "frames": True}), encoding="utf-8")
    monkeypatch.setattr("avm.core.LMU.exec", lambda self, conv, para: ("ok", [], conv))

    rc = main(["--debug-config", str(cfg_path)])
    assert rc == 0
    assert "监测帧" in capsys.readouterr().err


def test_main_checkpoint_in_flag(tmp_path, capsys, monkeypatch):
    path = _write(tmp_path, _no_init_image())
    cp = _make_checkpoint(tmp_path)
    monkeypatch.setattr("avm.core.LMU.exec", lambda self, conv, para: ("ok", [], conv))

    rc = main([path, "--checkpoint-in", cp])
    assert rc == 0


def _make_snapshot(tmp_path):
    """造一份与镜像匹配的重启快照（镜像没有 init，空核心直接拍）。"""
    from avm.core import Core
    from avm.image import load_image

    image_path = _write(tmp_path, _no_init_image())
    core = load_image(Core(), image_path)
    path = tmp_path / "snap.json"
    core.snapshot_to(str(path))
    return image_path, str(path)


def test_main_restore_snapshot(tmp_path, capsys):
    image_path, snap = _make_snapshot(tmp_path)
    rc = main([image_path, "--restore", snap])
    assert rc == 0
    assert "已从重启快照恢复" in capsys.readouterr().err


def test_main_writes_snapshot_on_clean_exit(tmp_path, capsys):
    image = _no_init_image()
    image["snapshot_to"] = "out/run.snapshot.json"
    path = _write(tmp_path, image)
    rc = main([path])
    assert rc == 0
    snap = tmp_path / "out" / "run.snapshot.json"
    assert snap.is_file()
    assert json.loads(snap.read_text(encoding="utf-8"))["format"] == "avm.snapshot"
    assert "重启快照已写入" in capsys.readouterr().err
