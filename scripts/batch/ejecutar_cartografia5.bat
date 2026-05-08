@echo off
setlocal EnableDelayedExpansion

REM =====================================================
REM  Cartografia pastoral v5 Aralar - Todos los años
REM  ACTUALIZADO Nivel 2 abril 2026:
REM    - Llama a cartografia_pastoral_v5.py
REM    - Pasa hasta 3 estrategias IPP adicionales A B D ademas de la C
REM    - Si no hay GeoTIFFs anuales para B y D, las espacializa por zona
REM      desde el CSV diagnostico_pastoral.csv del script R v9
REM    - Calcula consenso + concordancia automaticamente
REM    - Detecta automaticamente la banda modelo_inv del pipeline GEE dual
REM    - NUEVO v5: vuelca todo el resumen a aralar_resumen_YYYY.txt
REM =====================================================

set HIC=.\HIC_tipo_aralar.tif
set BORDAS=.\dist_borda_aralar.tif
set DEM=.\MDT05_aralar.tif
set PROSAIL_DIR=.\

REM CSV zonal del script R v9. Si existe, se usa para espacializar
REM B y D cuando no haya GeoTIFFs anuales propios.
set IPP_CSV=.\diagnostico_pastoral.csv
set IPP_CSV_ARG=
if exist %IPP_CSV% (
    set IPP_CSV_ARG=--ipp_csv %IPP_CSV%
    echo CSV zonal R encontrado: %IPP_CSV%
    echo   - Se espacializaran B y D por zona si no hay GeoTIFFs anuales propios
) else (
    echo AVISO: no existe %IPP_CSV%
    echo   - Solo se cartografiaran las estrategias con GeoTIFF anual propio
    echo   - Para activar B y D zonales, exportar el CSV desde R
)

REM =====================================================
REM  ESCENARIO 1: Carga real estimada 0.5 UGM/ha
REM =====================================================
set CARGA=0.5
set TEMPORADA=180

for %%Y in (2018 2019 2020 2021 2022 2023 2024 2025) do (
    echo.
    echo ============================================================
    echo   Procesando %%Y - Carga !CARGA! UGM/ha, !TEMPORADA! dias
    echo ============================================================
    if not exist mapas\%%Y mkdir mapas\%%Y

    REM Estrategia A - modelo G6 espacializado desde GEE
    set IPP_A_ARG=
    if exist !PROSAIL_DIR!\IPP_estrategia_A_%%Y.tif (
        set IPP_A_ARG=--ipp_a !PROSAIL_DIR!\IPP_estrategia_A_%%Y.tif
        echo   IPP A G6 disponible GeoTIFF para %%Y
    )

    REM Estrategia B - GeoTIFF si existe, si no se intentara desde CSV
    set IPP_B_ARG=
    if exist !PROSAIL_DIR!\IPP_estrategia_B_%%Y.tif (
        set IPP_B_ARG=--ipp_b !PROSAIL_DIR!\IPP_estrategia_B_%%Y.tif
        echo   IPP B disponible GeoTIFF para %%Y
    )

    REM Estrategia D - GeoTIFF si existe, si no se intentara desde CSV
    set IPP_D_ARG=
    if exist !PROSAIL_DIR!\IPP_estrategia_D_%%Y.tif (
        set IPP_D_ARG=--ipp_d !PROSAIL_DIR!\IPP_estrategia_D_%%Y.tif
        echo   IPP D disponible GeoTIFF para %%Y
    )

    python cartografia_pastoral_v5.py ^
        --prosail !PROSAIL_DIR!\PROSAIL_NN_Aralar_%%Y.tif ^
        --hic !HIC! ^
        --bordas !BORDAS! ^
        --dem !DEM! ^
        --output_dir mapas\%%Y ^
        --carga !CARGA! ^
        --temporada !TEMPORADA! ^
        !IPP_A_ARG! !IPP_B_ARG! !IPP_D_ARG! !IPP_CSV_ARG!
)

REM =====================================================
REM  ESCENARIO 2: Carga recomendada LIFE 0.7 UGM/ha
REM =====================================================
set CARGA=0.7

for %%Y in (2018 2019 2020 2021 2022 2023 2024 2025) do (
    echo.
    echo ============================================================
    echo   Procesando %%Y - Escenario LIFE !CARGA! UGM/ha
    echo ============================================================
    if not exist mapas\%%Y_LIFE mkdir mapas\%%Y_LIFE

    set IPP_A_ARG=
    if exist !PROSAIL_DIR!\IPP_estrategia_A_%%Y.tif (
        set IPP_A_ARG=--ipp_a !PROSAIL_DIR!\IPP_estrategia_A_%%Y.tif
    )
    set IPP_B_ARG=
    if exist !PROSAIL_DIR!\IPP_estrategia_B_%%Y.tif (
        set IPP_B_ARG=--ipp_b !PROSAIL_DIR!\IPP_estrategia_B_%%Y.tif
    )
    set IPP_D_ARG=
    if exist !PROSAIL_DIR!\IPP_estrategia_D_%%Y.tif (
        set IPP_D_ARG=--ipp_d !PROSAIL_DIR!\IPP_estrategia_D_%%Y.tif
    )

    python cartografia_pastoral_v5.py ^
        --prosail !PROSAIL_DIR!\PROSAIL_NN_Aralar_%%Y.tif ^
        --hic !HIC! ^
        --bordas !BORDAS! ^
        --dem !DEM! ^
        --output_dir mapas\%%Y_LIFE ^
        --carga !CARGA! ^
        --temporada !TEMPORADA! ^
        !IPP_A_ARG! !IPP_B_ARG! !IPP_D_ARG! !IPP_CSV_ARG!
)

echo.
echo ============================================================
echo   COMPLETADO
echo   Escenario 1 0.5 UGM/ha:  mapas\2018\ ... mapas\2025\
echo   Escenario 2 0.7 UGM/ha:  mapas\2018_LIFE\ ... mapas\2025_LIFE\
echo.
echo   En cada carpeta YYYY se ha generado:
echo     - PNG y GeoTIFF de los productos pastorales
echo     - aralar_resumen_YYYY.txt con todas las estadisticas
echo     - Mapas de IPP de las estrategias disponibles
echo     - aralar_mapa_IPP_consenso_pastos_YYYY.png  si hay al menos 3 estrategias
echo     - aralar_mapa_IPP_concordancia_pastos_YYYY.png  k/n por pixel
echo     - aralar_mapa_modelo_invertido_YYYY.png  si banda modelo_inv presente
echo ============================================================

endlocal
pause
