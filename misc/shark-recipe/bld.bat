mkdir build
cd build

:: Configure.
cmake -G"MinGW Makefiles" ^
      -DBUILD_EXAMPLES=OFF ^
      -DBUILD_DOCUMENTATION=OFF ^
      -DBUILD_SHARED_LIBS=ON ^
      -DCMAKE_BUILD_TYPE=Release ^
      %SRC_DIR%
if errorlevel 1 exit /b 1

:: Build.
make -j %{CPU_COUNT%
if errorlevel 1 exit /b 1

:: Test.
ctest -C Release
if errorlevel 1 exit /b 1