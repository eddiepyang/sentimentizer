from pathlib import Path
file_root = Path(__file__).parent.absolute()
relative_root = Path(".").resolve()

if __name__=="__main__":

    print("root is:", file_root, relative_root)
