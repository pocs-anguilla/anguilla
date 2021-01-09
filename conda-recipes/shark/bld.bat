mkdir build
cd build

:: Configure.
cmake -G"Ninja" ^
      -DBUILD_EXAMPLES=OFF ^
      -DBUILD_DOCUMENTATION=OFF ^
      -DBUILD_SHARED_LIBS=ON ^
      -DCMAKE_TOOLCHAIN_FILE=../cxx14.toolchain ^
      -DCMAKE_BUILD_TYPE=Release ^
      %SRC_DIR%
if errorlevel 1 exit /b 1

:: Build.
cmake --build .
if errorlevel 1 exit /b 1

:: Test.
ctest -C Release
if errorlevel 1 exit /b 1
