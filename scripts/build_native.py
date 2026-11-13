"""Reproduce the project-local OpenFace/R build on macOS Apple Silicon.

Installs dependencies under .runtime and pinned sources under external. Existing
runtimes are reused. No system-wide package installation is performed.
"""

import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess

from classroom_behaviour.native_tracking import run_logged

ROOT = Path(__file__).resolve().parents[1]
OPENFACE_REVISION = '3d4b5cf8d96138be42bed229447f36cbb09a5a29'
DLIB_REVISION = '39749fba5163d4e25030358f6aa8d91617bfdca2'


def clone_pinned(url, destination, revision):
    if not destination.exists():
        destination.mkdir(parents=True)
        for arguments in (['init'], ['remote','add','origin',url], ['fetch','--depth','1','origin',revision], ['checkout','--detach','FETCH_HEAD']):
            subprocess.run(['git','-C',str(destination),*arguments],check=True)
    actual=subprocess.check_output(['git','-C',str(destination),'rev-parse','HEAD'],text=True).strip()
    if actual!=revision:
        raise ValueError(f'{destination} is at {actual}; expected {revision}. Preserve it and use a matching checkout.')


def repair_r_paths(prefix):
    # Conda's R shell wrapper does not quote assignments containing spaces.
    path=prefix/'lib/R/bin/R'
    text=path.read_text()
    for key in ('R_HOME_DIR','R_SHARE_DIR','R_INCLUDE_DIR','R_DOC_DIR'):
        name=key+'='
        text='\n'.join(name+shlex.quote(line[len(name):])
                       if line.startswith(name) and ' ' in line and not line[len(name):].startswith(('"',"'"))
                       else line for line in text.split('\n'))
    path.write_text(text)


def main():
    if platform.system()!='Darwin' or platform.machine()!='arm64':
        raise RuntimeError('This build helper targets macOS arm64. See upstream OpenFace installation for other platforms.')
    conda=shutil.which('conda')
    if not conda:
        raise RuntimeError('Miniconda/Miniforge is required for isolated native dependencies')
    (ROOT/'logs/build').mkdir(parents=True, exist_ok=True)
    environment={**os.environ,'CONDA_PKGS_DIRS':str(ROOT/'.runtime/conda-pkgs')}
    for name,packages in [('r',['r-base=4.4','r-matrixcalc']),('openface',['cmake=3.31','libopencv=4.10','tbb-devel','openblas','boost-cpp'])]:
        prefix=ROOT/'.runtime'/name
        if not (prefix/'conda-meta').is_dir():
            subprocess.run([conda,'create','--prefix',str(prefix),'--override-channels','-c','conda-forge',*packages,'-y'],
                           env=environment,check=True)
    repair_r_paths(ROOT/'.runtime/r')
    clone_pinned('https://github.com/TadasBaltrusaitis/OpenFace.git',ROOT/'external/OpenFace',OPENFACE_REVISION)
    clone_pinned('https://github.com/davisking/dlib.git',ROOT/'external/dlib',DLIB_REVISION)
    dlib=ROOT/'external/dlib'
    compatibility=dlib/'dlib/global_optimization/find_max_global.h'
    compatibility.write_text(compatibility.read_text().replace('::template go(std::forward<T>(f),a)',
                                                               '::template go<T>(std::forward<T>(f),a)'))
    prefix=ROOT/'.runtime/openface';cmake=prefix/'bin/cmake'
    run_logged([cmake,'-S',dlib/'dlib','-B',ROOT/'external/dlib-build','-D','CMAKE_BUILD_TYPE=Release',
                '-D',f'CMAKE_INSTALL_PREFIX={prefix}','-D','DLIB_NO_GUI_SUPPORT=ON','-D','DLIB_USE_CUDA=OFF',
                '-D','DLIB_USE_BLAS=OFF','-D','DLIB_USE_LAPACK=OFF','-D','CMAKE_OSX_ARCHITECTURES=arm64'],ROOT/'logs/build/dlib_configure.log')
    run_logged([cmake,'--build',ROOT/'external/dlib-build','--parallel','4'],ROOT/'logs/build/dlib_build.log')
    run_logged([cmake,'--install',ROOT/'external/dlib-build'],ROOT/'logs/build/dlib_install.log')
    from scripts.setup_models import download_openface_models
    download_openface_models(ROOT/'external/OpenFace')
    build=ROOT/'external/OpenFace/build'
    run_logged([cmake,'-S',ROOT/'external/OpenFace','-B',build,'-D','CMAKE_BUILD_TYPE=Release',
                '-D',f'CMAKE_PREFIX_PATH={prefix}','-D',f'OpenBLAS_LIB={prefix}/lib/libopenblas.dylib',
                '-D',f'OpenBLAS_INCLUDE_DIR={prefix}/include','-D','CMAKE_OSX_ARCHITECTURES=arm64'],ROOT/'logs/build/openface_configure.log')
    run_logged([cmake,'--build',build,'--target','FeatureExtraction','--parallel','4'],ROOT/'logs/build/openface_build.log')
    print('Native tools ready. Run python -m scripts.setup_models --download-hmad, then python -m classroom_behaviour --check-dependencies.')


if __name__=='__main__':
    main()
