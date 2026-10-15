# Taxonomía de macro-categorías

Propuesta a partir del clustering exploratorio (`notebooks/01_eda_taxonomia.ipynb`) y de
una muestra aleatoria de 160 nombres. Hallazgo: k-means sobre n-gramas de caracteres
agrupa sobre todo por **marca** (ITELY, ALFAPARF, COQUETAS, MIA SECRET), con silhouette
de 0,02 a 0,04, y deja dos clústeres «cajón de sastre» con ~5 700 productos. Por eso la
taxonomía se define por **tipo de producto** con criterio humano y no se copia de los
clústeres.

Regla general: se clasifica por **para qué sirve el producto**, no por la marca. Una
herramienta específica de un dominio va a ese dominio (un pincel de uñas es «Uñas»).

| Categoría | Definición | Ejemplos reales |
|---|---|---|
| Uñas | Esmaltes, geles, acrílicos, monómeros, uñas postizas, decoración, limas, lámparas UV/LED y removedores para manicure/pedicure | ESMALTE RODHER TULIPAN · G.NAILS RUBBER BASE 14 UV/LED · ACRILICO COVER NUDE 0.5 OZ · UÑAS X500 MASTER NAILS · TIJERA PARA CUTICULA MIA SECRET |
| Cuidado capilar | Shampoo, acondicionador, mascarilla capilar, tratamientos, ampollas, aceites y óleos, termoprotectores, alisados/keratina, ceras y geles de peinado; incluye matizantes de cuidado (shampoo o mascarilla matizante) | VINILADY COCO NUTRICIÓN SHAMPOO 900ML · HAIRLISS AMPOLLA CAPILAR · PROKPIL COBRE TREATMENT · KIT DE KERATINA RENE CHARDON · HAIR WAX MATTE PASTE |
| Coloración capilar | Tintes, oxidantes/oxigentas, reveladores, decolorantes y colorantes de fantasía | TINTE SALONIN # 4 · ITELY AQUARELY 8C/8.1 · OXIGENTA # 10 SALONIN · GOLD´MAR POLVO DECOLORANTE · KUUL FUNNY COLORS TINTE CORAL |
| Maquillaje | Rostro, ojos y labios: bases, correctores, polvos, rubores, iluminadores, sombras, delineadores, labiales, primers y fijadores; incluye brochas y esponjas de maquillaje | L.A. COLORS CONCEALER · RUBOR SWEET STRIKE · BE BELLA PALETA DE SOMBRA · LABIAL YELLY · ESPONJA BLENDER X 6 |
| Pestañas y cejas | Pestañas postizas y de extensión, pegamentos, lifting y laminado, accesorios para su aplicación | SATINE LASH&GO PESTANA BANDA 3D · PESTAÑAS EN TIRA · LIBRO DE PESTAÑAS · BEAUTIFUL MOLDES PARA LIFTING · CALIBRADOR DE CEJAS |
| Piel y cuerpo | Skincare facial y corporal, limpiadores, mascarillas faciales, protector solar, depilación, fragancias y colonias | PONDS ACLARANTE PIEL GRASA · COSRX LOW PH GEL CLEANSER · BYPHASSE MASCARILLA FACIAL ARCILLA · CERA MIEL KALU · PERFUME ASAD LATTAFA |
| Herramientas y equipos de cabello | Secadoras, planchas, rizadores, máquinas de corte, cepillos, peines, tijeras y navajas, capas y papel de peluquería | SECADORA EXPERIENCE PROFESIONAL · PLANCHA REALISS EXPERT · KEMEI KM-510 · STAR T.PEIN.CARBON · PAPEL PELUQUERIA |
| Extensiones y accesorios de cabello | Cabello sintético y extensiones, pelucas, vinchas, chumis, invisibles, pinzas y rulos de peinado | CANECALON EZEBRAID RUBIO · CHUMIS PARA CABELLO # 2 · INVISIBLE DE BRONCE · RULO PLASTICO PARA CABELLO · VINCHA BOFEINI |
| Otros | Lo que no es belleza ni cuidado personal, o no encaja en ninguna anterior: bisutería, termos, juguetes, artículos varios | FASHION JEWELRY · TOMATODO KUROMI · MINI VENTILADOR F2 |

## Casos frontera

- **Matizantes:** el shampoo o la mascarilla matizante es «Cuidado capilar»; el
  pigmento o tinte de fantasía es «Coloración capilar».
- **Lápiz de cejas o sombra de cejas** es «Maquillaje»; **pestañas postizas, lifting
  y sus herramientas** es «Pestañas y cejas».
- **Brochas y pinceles:** de maquillaje → «Maquillaje»; de uñas → «Uñas»; brocha
  para aplicar tinte → «Coloración capilar».
- **Pinzas:** de pestañas o cejas → «Pestañas y cejas»; de peinado → «Extensiones y
  accesorios de cabello».
- **Kits o combos mixtos:** según el producto principal del nombre.
- Si el nombre no permite decidir con razonable seguridad → «No sé» (se excluye).
