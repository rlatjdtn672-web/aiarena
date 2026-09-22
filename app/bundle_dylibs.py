#!/usr/bin/env python3
"""엔진 실행 파일이 쓰는 라이브러리를 앱 안으로 옮긴다.

    python bundle_dylibs.py <실행파일> <lib 폴더> [추가로 넣을 dylib ...]

homebrew 에 깔린 SDL2 등을 그대로 참조하면 다른 맥에서는 안 뜬다.
딸린 라이브러리를 줄줄이 따라가며 lib 폴더로 복사하고, 서로를 가리키는 경로를
"@loader_path/..." 로 바꾼 뒤 다시 서명한다 (애플 실리콘은 서명이 깨지면 실행을 막는다).

추가 dylib 인자는 코드가 dlopen 으로 여는 것들이다 — otool 로는 안 보인다.
sdl2-compat 가 여는 libSDL3 가 그렇다.
"""
import os
import shutil
import subprocess
import sys

EXTERNAL = ("/opt/homebrew/", "/usr/local/")


def deps(path):
    out = subprocess.run(["otool", "-L", path], capture_output=True, text=True, check=True).stdout
    res = []
    for line in out.splitlines()[1:]:
        p = line.strip().split(" (")[0]
        if p.startswith(EXTERNAL) or p.startswith("@rpath/"):
            res.append(p)
    return res


def resolve(ref, origin):
    """참조 경로를 실제 파일로. @rpath 는 homebrew lib 폴더들에서 찾는다."""
    if ref.startswith("@rpath/"):
        leaf = ref.split("/", 1)[1]
        for base in (os.path.dirname(origin), "/opt/homebrew/lib"):
            cand = os.path.join(base, leaf)
            if os.path.exists(cand):
                return os.path.realpath(cand)
        return None
    return os.path.realpath(ref) if os.path.exists(ref) else None


def run(*a):
    subprocess.run(a, check=True, capture_output=True)


def main():
    exe, libdir, *extra = sys.argv[1:]
    os.makedirs(libdir, exist_ok=True)
    copied = {}                 # 원래 참조 경로 → lib 안 파일 이름
    todo = [(exe, True)]
    for e in extra:
        leaf = os.path.basename(e)
        dst = os.path.join(libdir, leaf)
        shutil.copy2(os.path.realpath(e), dst)
        os.chmod(dst, 0o755)
        copied[e] = leaf
        todo.append((dst, False))

    seen = set()
    while todo:
        path, is_exe = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        for ref in deps(path):
            real = resolve(ref, path)
            if real is None:
                print(f"  ! 못 찾음: {ref}  (from {os.path.basename(path)})")
                continue
            leaf = os.path.basename(ref)
            dst = os.path.join(libdir, leaf)
            if not os.path.exists(dst):
                shutil.copy2(real, dst)
                os.chmod(dst, 0o755)
                todo.append((dst, False))
            new = "@loader_path/lib/" + leaf if is_exe else "@loader_path/" + leaf
            run("install_name_tool", "-change", ref, new, path)
        if not is_exe:
            run("install_name_tool", "-id", "@loader_path/" + os.path.basename(path), path)

    for p in list(seen):
        run("codesign", "--force", "-s", "-", p)
    libs = sorted(os.listdir(libdir))
    print(f"라이브러리 {len(libs)}개를 옮겼습니다:")
    for n in libs:
        print("  ", n)


if __name__ == "__main__":
    main()
