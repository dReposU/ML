# ML
Repositorio del curso de Machine Learning (1ACC0057) de la UPC del ciclo 2026-01. Material del 2026-01 [aquí](https://drive.google.com/drive/u/1/folders/1lf9_RwgqqbKHlgZh5rOZp1U2Gqu2ThwR).

# Abrir archivos .ipynb en Google Colab
Para abrir un archivo `.ipynb` en Google Colab, reemplaza `github.com` por `colab.research.google.com/github/` en la URL del repositorio.

Puedes buscar el archivo en el repo y copiar el URL del buscador.

**Ejemplo:**
```
https://github.com/dReposU/ML/blob/main/Sem_1/Ejercicio-Preprocesamiento/01_ML_Preprocesamiento_CriptoMonedas.ipynb
->
https://colab.research.google.com/github/dReposU/ML/blob/main/Sem_1/Ejercicio-Preprocesamiento/01_ML_Preprocesamiento_CriptoMonedas_mod.ipynb
```

# Convenciones de nombre

Si al final del nombre del `.ipynb` está escrito `clase` u `original`, se suele referir a que el archivo sigue estrictamente el código y resultados del profesor. Con el único cambio el código de carga de datos para que sea compatible en VSC y Google Colab:

```python
import sys

if 'google.colab' in sys.modules:
    data = pd.read_csv(f'https://raw.githubusercontent.com/dReposU/ML/main/.../foo.csv')
else:
    data = pd.read_csv(".../foo.csv")
data.head()
```
