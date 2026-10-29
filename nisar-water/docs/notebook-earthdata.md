# Notebook público con token Earthdata propio

Abre [NISAR_Water_Try_Your_Data.ipynb](../notebooks/NISAR_Water_Try_Your_Data.ipynb)
en Jupyter o en
[Google Colab](https://colab.research.google.com/github/Ch0comilo/NISAR-Models/blob/main/nisar-water/notebooks/NISAR_Water_Try_Your_Data.ipynb).
El notebook se distribuye sin ejecuciones ni salidas y no contiene credenciales.

## Inicio local

```sh
git clone https://github.com/Ch0comilo/NISAR-Models.git
cd NISAR-Models/nisar-water
uv sync --frozen --extra notebook --python 3.13
uv run --no-sync jupyter lab notebooks/NISAR_Water_Try_Your_Data.ipynb
```

Usa Python 3.12 o 3.13. En Colab, la primera celda clona el repositorio cuando
hace falta y prepara un intérprete de inferencia aislado para evitar conflictos
con NumPy u otras librerías del kernel. No exige reiniciar el kernel. La
preparación inicial necesita conexión y puede tardar varios minutos.

## Qué se configura

- `BBOX`: oeste, sur, este, norte, en longitud/latitud WGS84.
- `START_DATE`, `END_DATE`: intervalo de fechas UTC de hasta 366 días.
- `MODEL_PROFILE`: `radar`, `terrain` o `terrain_prior`. El último es el perfil
  conservador de comparación; ninguno está aceptado para uso general.
- `MAX_OBSERVATIONS`: 2 limita el ejemplo; 0 solicita todas las compatibles
  dentro del presupuesto. El estado partial identifica una consulta incompleta.
- `TRANSFER_CAP_MIB`: presupuesto total de radar, con máximo 128 MiB por
  adquisición. El contexto físico público tiene un presupuesto independiente
  de hasta 128 MiB por región y se conserva en caché.

La zona debe ser pequeña: máximo 0,2° por lado incluido el contexto y cuatro
millones de píxeles nativos. El lector admite el perfil NISAR L-GCOV provisional
v1, frecuencia A, HH/HV en potencia gamma0 lineal, con grilla UTM nativa de
10/20 m. Colecciones/perfiles nuevos necesitan verificación y adaptación; no se
sustituyen silenciosamente. La selección no depende de nubes, óptica ni del
resultado del modelo. Las series separan track, paso, frame y frecuencia.

## Token y privacidad

Genera/obtén un token Bearer en tu propia cuenta de
[Earthdata Login](https://urs.earthdata.nasa.gov/). Debes tener el acceso y los
acuerdos de datos exigidos por NASA/ASF. Pégalo únicamente en el campo de
`getpass`. Se admite el prefijo `Bearer ` opcional.

También se acepta `EARTHDATA_TOKEN` para ejecución automatizada. El notebook lo
retira del entorno del kernel al leerlo y lo suministra solo al proceso de
inferencia. El valor no está en argumentos del comando, URLs, archivos de
configuración ni salidas. La variable del notebook se vacía tras la inferencia.
La búsqueda pública CMR no recibe el token; las redirecciones a hosts públicos
eliminan Authorization. Los recibos guardan categorías, hosts, bytes y hashes.

No guardes un token en celdas, capturas ni commits. Si ejecutas el notebook,
limpia sus salidas antes de proponer un commit; la verificación del repositorio
exige conservar limpio el notebook público.

## Resultados y API

El ejemplo usa observaciones de Nam Theun 1 del 10 y 22 de octubre de 2026.
Muestra la composición radar/agua, incertidumbre, área de agua en soporte común,
ganancia/pérdida y una animación cuando las vistas coinciden. No es otra prueba
independiente de precisión. La zona ya formó parte de evaluaciones anteriores.

La última celda inicia un servidor temporal en localhost, consulta salud,
assessment y series, y verifica una imagen por HTTP y SHA256. Se detiene al
terminar. El servidor existente publica resultados calculados; los endpoints
de envío de trabajos de inferencia continúan siendo una integración futura.

Los archivos científicos quedan bajo `outputs/earthdata-demo-*`. Máscaras:
0 no agua, 1 agua, 254 desconocido, 255 no observado. Contexto requerido ausente
es desconocido. No interpretes cambios de cobertura como cambios del agua ni
atribuyas actividad humana a partir de esta clasificación.

## Errores frecuentes

| Categoría | Acción |
|---|---|
| `AUTHENTICATION_FAILED` | Revisar o renovar el token propio |
| `AUTHORIZATION_OR_AGREEMENT_REQUIRED` | Revisar permisos/acuerdos Earthdata/ASF |
| `TRANSFER_LIMIT` | Reducir la zona/fechas o ampliar explícitamente el presupuesto |
| `no_compatible_data` | Cambiar zona/fechas; el perfil disponible puede no ser el admitido |
| `RANGE_NOT_SUPPORTED` | El servidor no confirmó el rango; nunca se descarga todo como alternativa |
| `NETWORK_FAILURE` | Reintentar con conexión al servicio; conservar el estado partial |
| Mucho desconocido | Revisar soporte radar, contexto físico y límites del modelo |

El lector independiente vive en `remote_gcov.py`, `earthdata.py` y
`live_acquisition.py`. Los campos de potencia, validez y recorte conservan el
enfoque del lector local original; el módulo nuevo verifica explícitamente
identidad de producto, fecha, grilla y límites. Los checkpoints y fuentes
científicas congeladas no cambian.

## Verificación realizada

El notebook se ejecutó completo el 29 de octubre de 2026: siete celdas de código,
15 salidas y cero errores. Adquirió y procesó las dos fechas reales indicadas,
con 32.505.872 bytes de cuerpos de rangos radar (aproximadamente 31 MiB), y
verificó COG, vectores, áreas, transparencia y activos HTTP. La adquisición de
contexto público es adicional. El ejemplo quedó completo para la consulta
compatible; no se midió precisión ni se modificó el modelo.

El [recibo público sin credenciales](../benchmarks/notebook-online-demo.json)
conserva identidades, soporte y comprobaciones. La ejecución real se hizo en
Jupyter/Python 3.13; la interfaz alojada de Colab no se probó manualmente. El
notebook publicado se mantiene limpio para que cada usuario lo ejecute con
su propia cuenta y configuración.
