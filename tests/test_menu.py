import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

spec = importlib.util.spec_from_file_location('launcher', Path(__file__).resolve().parents[1]/'main.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.mark.parametrize('choice,module', [('1','bounding_box.pipeline'),('2','hog_svm.pipeline'),('3','yolov8.pipeline')])
def test_menu_routes_each_method(choice, module, monkeypatch):
    answers = iter([choice, '1', 'exit'])
    calls = []
    monkeypatch.setattr('builtins.input', lambda _: next(answers))
    monkeypatch.setattr(launcher.importlib, 'import_module', lambda name: (calls.append(name) or SimpleNamespace(train=lambda: calls.append('trained'))))
    launcher.main()
    assert calls == [module, 'trained']


def test_invalid_choice_and_missing_model_return_to_menu(monkeypatch, capsys):
    answers = iter(['wrong','2','2','exit'])
    monkeypatch.setattr('builtins.input', lambda _: next(answers))
    def missing():
        raise FileNotFoundError('Train first')
    monkeypatch.setattr(launcher.importlib, 'import_module', lambda _: SimpleNamespace(evaluate=missing))
    launcher.main()
    output=capsys.readouterr().out
    assert 'Please enter' in output and 'Train first' in output


def test_menu_opens_comparison_without_running_a_method(monkeypatch):
    answers=iter(['4','exit'])
    calls=[]
    monkeypatch.setattr('builtins.input',lambda _:next(answers))
    monkeypatch.setattr(launcher.importlib,'import_module',lambda name: (calls.append(name) or SimpleNamespace(display=lambda root:calls.append(root))))
    launcher.main()
    assert calls==['common.comparison',launcher.ROOT]
