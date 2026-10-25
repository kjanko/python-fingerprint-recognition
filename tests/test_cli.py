import json

import cv2

from fingerprint_recognition.cli import main


def test_matching_prints_exit_zero(database, capsys):
    code = main([str(database / "101_1.tif"), str(database / "101_2.tif")])

    assert code == 0
    assert "Fingerprint matches." in capsys.readouterr().out


def test_different_prints_exit_one(database, capsys):
    code = main([str(database / "101_1.tif"), str(database / "102_1.tif")])

    assert code == 1
    assert "does not match" in capsys.readouterr().out


def test_json_output(database, capsys):
    main([str(database / "102_1.tif"), str(database / "102_2.tif"), "--json"])

    result = json.loads(capsys.readouterr().out)
    assert result["is_match"] is True
    assert result["score"] >= result["threshold"] == 12
    assert len(result["minutiae"]) == 2


def test_save_writes_the_match_image(database, tmp_path):
    output = tmp_path / "matches.png"

    main([str(database / "101_1.tif"), str(database / "101_2.tif"), "--save", str(output)])

    assert cv2.imread(str(output)) is not None


def test_missing_file_exits_two(database, tmp_path, capsys):
    code = main([str(database / "101_1.tif"), str(tmp_path / "missing.tif")])

    assert code == 2
    assert "Could not read image" in capsys.readouterr().err
