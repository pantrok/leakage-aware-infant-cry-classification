# Diagnóstico de agrupamiento por grabación

Diagnóstico puro — **no se modificó `_recording_id()` ni se relanzó
`leakage_comparison.py`** en esta pasada, según lo pedido.

Nota sobre datos: no se re-copió nada a `data/dataset/` (que está vacío,
gitignored) — todo esto se corrió directo contra los `.wav` reales en
`../Bebes Grid indiv - Optuna - 2/Data/` (la copia local original, fuera
del repo git), que es donde ya sabíamos que viven.

## Resumen ejecutivo (para no tener que leer todo)

Encontré evidencia dura — no solo un patrón visual — de que el nombre de
archivo de 10 dígitos es `PPPP SSS CCC`: 4 dígitos de **ID de grabación**,
3 dígitos de **contador de segmento** (reinicia en `001` en cada
grabación nueva) y 3 dígitos de **código constante por clase**. La
"Tarea 2" encontró 6 archivos `.doc` de 2002 (logs de procesamiento
originales del dataset, no generados por el código de este repo) dentro
de `Data/1s_deaf/`, cuyo conteo de líneas coincide **exactamente**,
archivo por archivo, con el conteo real de `.wav` por prefijo de 4
dígitos en esa carpeta — es la confirmación más fuerte posible de que el
prefijo de 4 dígitos sí es el ID real de grabación. Con ese criterio, el
total estimado de grabaciones distintas (`M`) sería del orden de **73**
en las 5 clases combinadas, muy lejos de las 2268 "grabaciones" que
reporta `_recording_id()` tal como está hoy.

---

## Tarea 1 — Muestra de nombres (1s_X) por clase

```
asphyxia (340 archivos totales):
  0063001030.wav
  0063002030.wav
  0063003030.wav
  0063004030.wav
  0063005030.wav
  0063006030.wav
  0063007030.wav
  0063008030.wav
  0063009030.wav
  0063010030.wav
  0063011030.wav
  0063012030.wav
  0063013030.wav
  0063014030.wav
  0063015030.wav
  0063016030.wav
  0063017030.wav
  0063018030.wav
  0063019030.wav
  0063020030.wav

deaf (879 archivos .wav totales):
  0025001010.wav
  0025002010.wav
  0025003010.wav
  0025004010.wav
  0025005010.wav
  0025006010.wav
  0025007010.wav
  0025008010.wav
  0025009010.wav
  0025010010.wav
  0025011010.wav
  0025012010.wav
  0025013010.wav
  0025014010.wav
  0025015010.wav
  0025016010.wav
  0025017010.wav
  0025018010.wav
  0025019010.wav
  0025020010.wav

hunger (350 archivos totales):
  0004001001.wav
  0004002001.wav
  0004003001.wav
  0004004001.wav
  0004005001.wav
  0004006001.wav
  0004007001.wav
  0005001001.wav
  0005002001.wav
  0005003001.wav
  0011001001.wav
  0011002001.wav
  0011003001.wav
  0011004001.wav
  0011005001.wav
  0011006001.wav
  0012001001.wav
  0012002001.wav
  0012003001.wav
  0012004001.wav

normal (507 archivos totales):
  0052001000.wav
  0052002000.wav
  0052003000.wav
  0052004000.wav
  0052005000.wav
  0052006000.wav
  0052007000.wav
  0052008000.wav
  0052009000.wav
  0052010000.wav
  0052011000.wav
  0052012000.wav
  0052013000.wav
  0052014000.wav
  0052015000.wav
  0052016000.wav
  0052017000.wav
  0052018000.wav
  0052019000.wav
  0052020000.wav

pain (192 archivos totales):
  0001001002.wav
  0001002002.wav
  0001003002.wav
  0001004002.wav
  0001005002.wav
  0001006002.wav
  0001007002.wav
  0001008002.wav
  0002001002.wav
  0002002002.wav
  0002003002.wav
  0002004002.wav
  0002005002.wav
  0002006002.wav
  0002007002.wav
  0002008002.wav
  0003001002.wav
  0003002002.wav
  0003003002.wav
  0003004002.wav
```

**Patrón visual observado** (descriptivo, no es una regla que haya
aplicado en código): los primeros 4 dígitos se repiten entre archivos
consecutivos y luego saltan a un valor nuevo (ver `hunger`: `0004`×7 →
`0005`×3 → `0011`×6 → `0012`×... — saltos claros de "grabación"). Los
siguientes 3 dígitos suben de forma consecutiva 001, 002, 003... dentro
de cada bloque de 4 dígitos repetido, y vuelven a 001 al cambiar el
bloque — exactamente el comportamiento de un contador de segmento por
grabación. Los últimos 3 dígitos son **constantes dentro de cada clase**
en las muestras que tomé (`030` en asphyxia, `010` en deaf, `001` en
hunger, `000` en normal, `002` en pain) — parecen un código de
clase/sesión, no parte del ID de grabación ni del contador.

Como cross-check cuantitativo (conteo real sobre TODOS los `.wav`, no
solo la muestra de 20), agrupando únicamente por los primeros 4 dígitos:

| Clase | Archivos .wav | Prefijos de 4 dígitos únicos | Sufijo de 3 dígitos observado |
|---|---:|---:|---|
| asphyxia | 340 | 6 | `030` (constante) |
| deaf | 879 | 6 | `010` (constante) |
| hunger | 350 | 33 | `001` (constante) |
| normal | 507 | 5 | `000` (constante) |
| pain | 192 | 23 | `002` (constante) |
| **Total** | **2268** | **73** | — |

## Tarea 2 — Documentación/metadata encontrada

**Sí se encontró algo — y es fuerte.** Dentro de
`Bebes Grid indiv - Optuna - 2/Data/1s_deaf/` hay 6 archivos `.doc`
(no `.wav`) que no forman parte del código ni de este repo:

```
more intensity25.doc   (65024 bytes,  creado 2002-06-25)
more intensity26.doc   (49664 bytes,  creado 2002-06-25)
more intensity27.doc   (112128 bytes, creado 2002-06-25)
more intensity28.doc   (159744 bytes, creado 2002-06-25)
more intensity29.doc   (58880 bytes,  creado 2002-06-26)
more intensity30.doc   (100864 bytes, creado 2002-07-18)
```

Metadata del archivo (Word 97, extraída del OLE header):
`Author: Jos`, `Template: Normal.dot`, `Last Saved By: Jos`,
`Creating Application: Microsoft Word 9.0` — consistentes con
documentos de trabajo originales de 2002, muy anteriores a este
proyecto (no fueron generados por ningún script de este repo).

Contenido (extraído con `antiword`): son **logs de una herramienta de
procesamiento por lotes** (probablemente un script/macro que aplicó
pre-énfasis y volvió a escribir los WAV), con esta forma repetida:

```
Read from file... D:\Infant Cry Tesis\Llantos\Allcries\sordos\1
seg\0025001010.wav
Pre-emphasize... 50
Write to WAV file... D:\Infant Cry Tesis\Llantos\Allcries\sordos\1 seg
balance\0025001010.wav
Read from file... D:\Infant Cry Tesis\Llantos\Allcries\sordos\1
seg\0025002010.wav
Pre-emphasize... 50
Write to WAV file... D:\Infant Cry Tesis\Llantos\Allcries\sordos\1 seg
balance\0025002010.wav
... (se repite el patrón Read/Pre-emphasize/Write para cada archivo)
```

(`sordos` = "deaf" en español — confirma que esta carpeta corresponde a
la clase `deaf`/`sordos`.)

**Cross-check que confirma el patrón de la Tarea 1 con evidencia dura**
(no es solo un parecido visual): conté cuántas entradas "Read from
file..." tiene cada `.doc`, y lo comparé contra el conteo real de
`.wav` que empiezan con ese mismo prefijo de 4 dígitos en la carpeta
`1s_deaf` actual:

| Archivo .doc | Entradas "Read from file" | Prefijo | Archivos .wav reales con ese prefijo |
|---|---:|---|---:|
| more intensity25.doc | 121 | 0025 | 121 |
| more intensity26.doc | 56  | 0026 | 56  |
| more intensity27.doc | 158 | 0027 | 158 |
| more intensity28.doc | 234 | 0028 | 234 |
| more intensity29.doc | 70  | 0029 | 70  |
| more intensity30.doc | 240 | 0030 | 240 |

**Coincidencia exacta en los 6 casos.** Esto no es circunstancial: cada
`.doc` es literalmente el log de procesamiento de una grabación
completa, y el número de segmentos de 1s que hoy están en `1s_deaf` con
ese prefijo es idéntico al número de líneas de ese log. El prefijo de 4
dígitos SÍ es el identificador real de grabación, al menos para la
clase `deaf`.

No se encontraron archivos `.doc`/`.txt`/`.csv`/`.xlsx`/`readme*`/
`*codebook*` equivalentes en ninguna otra carpeta de clase
(`1s_asphyxia`, `1s_hunger`, `1s_normal`, `1s_pain`) ni en la raíz de
`Data/` o de `Bebes Grid indiv - Optuna - 2/` — solo aparecieron estos 6
en `1s_deaf/`. Por qué sobrevivieron solo ahí (y no para las otras 4
clases) no lo sé; puede ser simple azar de qué se conservó al copiar el
dataset originalmente.

## Tarea 3 — Comparación 1s_X vs Full_X

**No se encontró ninguna carpeta `Full_X` ni `FullNormal` en los datos
disponibles localmente.** `Data/` solo contiene las 5 carpetas `1s_X`
(confirmado también en la ronda 1 de este proyecto) — no hay nada que
comparar directamente.

Dato relacionado que sí apareció en los logs `.doc` de la Tarea 2: las
rutas originales de 2002 son
`D:\Infant Cry Tesis\Llantos\Allcries\sordos\1 seg\...` y
`...\1 seg balance\...` — es decir, la fuente que este log procesa **ya
está bajo una carpeta llamada "1 seg"**, no bajo algo llamado "Full". No
hay ninguna mención a "Full" en el contenido de estos logs. Esto no
prueba que `Full_X` nunca haya existido (el log solo documenta el paso
de pre-énfasis sobre segmentos ya cortados a 1s, no el paso de corte en
sí), pero sí es consistente con que en esta copia del dataset nunca
hubo carpetas `Full_X` — el corte a 1 segundo pudo haberse hecho en otro
momento/máquina no capturado por estos logs, o `Full_X` puede ser
terminología que el README del proyecto agregó después sin que la copia
de datos disponible la incluya.

## Tarea 4 — Conteo de archivos por carpeta Full_X

No aplica — no existe ninguna carpeta `Full_X`/`FullNormal` en los datos
disponibles localmente (ver Tarea 3), así que no hay nada que contar ni
comparar contra los 340/879/350/507/192 segmentos de 1s por clase.

Como referencia cruzada alternativa (ya que no hay `Full_X` con qué
comparar), sí se puede usar el conteo de prefijos únicos de 4 dígitos de
la Tarea 1 como estimado directo de "número de grabaciones reales" por
clase — validado con evidencia dura solo para `deaf` (Tarea 2), pero el
mismo patrón de nombre de archivo (10 dígitos, prefijo repetido +
contador que reinicia + sufijo constante por clase) se observa
visualmente en las 5 clases:

| Clase | Segmentos 1s (N) | Prefijos de 4 dígitos únicos (candidato a M) |
|---|---:|---:|
| asphyxia | 340 | 6 |
| deaf | 879 | 6 (confirmado con logs .doc) |
| hunger | 350 | 33 |
| normal | 507 | 5 |
| pain | 192 | 23 |
| **Total** | **2268** | **73** |
