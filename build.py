import re
import os
import zipfile
import pathlib


class SourceFlattener:
    def __init__(self, path):
        self.path = path
        self.imported = set()
        self.source = None

    def flatten(self):
        source = self.get_source(self.path)
        imports, source = self.resolve_imports(source)
        self.source = self._remove_multilines(self._arrange_imports(imports) + source)

    @staticmethod
    def _arrange_imports(imports):
        imports = [i for i in imports if i.strip() != ""]
        imports = sorted(imports, key=lambda x: (-ord(x[0]), len(x.split(" ")[1].split(".")[0]), len(x)))
        return SourceFlattener._append_lines(imports)

    @staticmethod
    def _remove_multilines(source):
        n = 0
        results = []
        for l in source:
            if l.strip() == "":
                n += 1
            else:
                n = 0
            if n <= 2:
                results.append(l)
        while results[-1].strip() == "":
            results = results[:-1]
        return results

    def publish(self, to_path):
        with open(to_path, "w") as file:
            file.writelines(self.source)

    def get_source(self, path):
        with open(path, "r") as file:
            return file.readlines()

    def parse_import(self, line):
        result = {"module": None, "objects": None, "path": None}
        match = re.match("^from .*", line)
        if match:
            result["module"] = line.split(" ")[1]
            result["objects"] = line.split(" import ")[1].split(", ")
            if result["module"].startswith("riiid"):
                result["path"] = "./" + "/".join(result["module"].split(".")) + ".py"
            return result

        match = re.match("^import .*", line)
        if match:
            result["module"] = line.split(" ")[1]
            result["objects"] = []
            return result

        return None

    def resolve_imports(self, source):
        imports = []
        sources = []
        for line in source:
            i = self.parse_import(line.strip())
            if i is not None:
                if i["path"] is not None:
                    if i["module"] not in self.imported:
                        self.imported.add(i["module"])
                        import_source = self.get_source(i["path"])
                        import_imports, import_source = self.resolve_imports(import_source)
                        imports.extend(import_imports)
                        sources.extend(import_source)
                elif line not in self.imported:
                    self.imported.add(line)
                    imports.append(line)
            else:
                sources.append(line)
        return self._append_lines(imports), self._append_lines(sources)

    @staticmethod
    def _append_lines(l):
        l.append("\n")
        l.append("\n")
        return l


def flatten_sources():
    builder = SourceFlattener("riiid/submit.py")
    builder.flatten()
    builder.publish("./riiid-submit.py")

    builder = SourceFlattener("./riiid/train.py")
    builder.flatten()
    builder.publish("./riiid-train.py")


def get_module_path():
    return str(pathlib.Path(__file__).parent)


def zip_package(path, model_path, model_name):
    import io

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip:
        for root, dirs, files in os.walk(path):
            for file in files:
                arcname = os.path.join("riiid", os.path.relpath(os.path.join(root, file), path))
                zip.write(os.path.join(root, file), arcname=arcname)
        zip.write(os.path.join(model_path, model_name), arcname="model.pkl")
    return zip_buffer


if __name__ == "__main__":
    from riiid.config import MODELS_PATH, SUBMIT_PATH

    model_name = "model_20201025_191555.pkl"

    zip_buffer = zip_package("./riiid", MODELS_PATH, model_name)

    zip_name = model_name.replace("pkl", "zip")
    with open(os.path.join(SUBMIT_PATH, zip_name), "wb") as zip_file:
        zip_file.write(zip_buffer.getvalue())
