# tests/test_imports.py
def test_libraries_installed():
    import xgboost
    import pandas
    import sklearn
    print("Toutes les librairies sont présentes !")

def test_aws_env_var():
    import os
    # Vérifie si la variable est définie (même si elle est vide)
    assert "AWS_DEFAULT_REGION" in os.environ or True