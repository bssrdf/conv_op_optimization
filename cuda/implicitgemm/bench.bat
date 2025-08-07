@echo off
echo ##### Start test #####
echo  N  H  W  C R S  K

for /L %%k in (1,1,2) do (
    for /L %%c in (1,1,2) do (
        for /L %%s in (1,1,2) do (
            set /a C=%%c*32
            set /a H=%%s*64
            set /a W=%%s*64
            set /a K=%%k*128
            rem Need delayed expansion to access updated variables
            call :run_implgemm
        )
    )
)

echo ##### Test finish! #####
pause
goto :eof

:run_implgemm
rem Enable delayed expansion so we can see updated variables
setlocal enabledelayedexpansion
.\implgemm 8 !C! !H! !W! !K! 3 3 1 1 0 0
endlocal
goto :eof
