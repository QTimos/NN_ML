import os
import sys

def setup(path: str):
    os.system(f"python3 -m venv {path}/NN_LM_venv")

def main() -> None:
    goinfre_path = f"/home/{os.getenv('USER')}/goinfre"
    curr_dir_path = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(goinfre_path):
        path_to_use = goinfre_path
    else:
        path_to_use = curr_dir_path
    setup(path_to_use)
    with open(".TMP.txt", "w") as f:
        f.write(f"{path_to_use}/NN_LM_venv/bin/activate")


if __name__ == "__main__":
    main()
