# NISAR Models

Clasificación experimental de agua en datos científicos NISAR, sin frontend,
con mapas coloreados, bordes y evolución temporal. El proyecto está dentro de
[`nisar-water/`](nisar-water/).

## Pruébalo con tu propio token Earthdata

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ch0comilo/NISAR-Models/blob/main/nisar-water/notebooks/NISAR_Water_Try_Your_Data.ipynb)

Abre [el notebook](nisar-water/notebooks/NISAR_Water_Try_Your_Data.ipynb), ejecuta
las celdas en orden e introduce **tu propio token Bearer de NASA Earthdata**
en el campo oculto. Puedes cambiar zona, fechas y perfil del modelo. El token
no se incluye en archivos, salidas, URLs ni commits.

Para Jupyter local:

```sh
git clone https://github.com/Ch0comilo/NISAR-Models.git
cd NISAR-Models/nisar-water
uv sync --frozen --extra notebook
uv run --no-sync jupyter lab notebooks/NISAR_Water_Try_Your_Data.ipynb
```

El ejemplo busca dos observaciones compatibles de Nam Theun 1, Laos; lee
recortes mediante rangos HTTPS; muestra agua/borde, soporte desconocido y
cambios temporales; y verifica el contrato HTTP para la futura interfaz.
No descarga gránulos completos ni necesita los proyectos locales anteriores.

**Estado del modelo: experimental, `not_ready_for_general_use`.** La inferencia
y el software funcionan; quedan fallos importantes de detección. Este notebook
no reentrena el modelo ni mide precisión. Cada persona necesita su propio
acceso Earthdata/ASF y los acuerdos de datos correspondientes.

- [Instalación, modelos y comandos](nisar-water/README.md)
- [Guía del notebook](nisar-water/docs/notebook-earthdata.md)
- [Arquitectura](nisar-water/docs/architecture.md)
- [Integración por API](nisar-water/docs/api-integration.md)
- [Tarjeta de los modelos](nisar-water/checkpoints/README.md)
