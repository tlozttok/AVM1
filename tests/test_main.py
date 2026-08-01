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
