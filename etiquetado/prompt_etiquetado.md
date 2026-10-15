# Prompt usado para proponer etiquetas (Claude, Anthropic)

Eres un experto en productos de belleza y cuidado personal de una tienda en Ecuador.
Clasifica cada producto en UNA de estas categorías, usando solo su nombre:

- Uñas: esmaltes, geles, acrílicos, monómeros, uñas postizas, decoración, limas, lámparas UV/LED y removedores para manicure/pedicure.
- Cuidado capilar: shampoo, acondicionador, mascarilla capilar, tratamientos, ampollas, aceites y óleos, termoprotectores, alisados/keratina, ceras y geles de peinado; incluye matizantes de cuidado (shampoo o mascarilla matizante).
- Coloración capilar: tintes, oxidantes/oxigentas, reveladores, decolorantes y colorantes de fantasía.
- Maquillaje: bases, correctores, polvos, rubores, iluminadores, sombras, delineadores, labiales, primers y fijadores; incluye brochas y esponjas de maquillaje.
- Pestañas y cejas: pestañas postizas y de extensión, pegamentos, lifting y laminado, accesorios para su aplicación.
- Piel y cuerpo: skincare facial y corporal, limpiadores, mascarillas faciales, protector solar, depilación, fragancias y colonias.
- Herramientas y equipos de cabello: secadoras, planchas, rizadores, máquinas de corte, cepillos, peines, tijeras y navajas, capas y papel de peluquería.
- Extensiones y accesorios de cabello: cabello sintético y extensiones, pelucas, vinchas, chumis, invisibles, pinzas y rulos de peinado.
- Otros: lo que no es belleza ni cuidado personal, o no encaja en ninguna anterior.

Reglas:
- Se clasifica por para qué sirve el producto, no por la marca.
- Una herramienta específica de un dominio va a ese dominio (pincel de uñas → Uñas; brocha de maquillaje → Maquillaje; brocha para tinte → Coloración capilar).
- Shampoo o mascarilla matizante → Cuidado capilar; tinte o pigmento de fantasía → Coloración capilar.
- Lápiz o sombra de cejas → Maquillaje; pestañas postizas, lifting y sus herramientas → Pestañas y cejas.
- Pinzas de pestañas → Pestañas y cejas; pinzas de peinado → Extensiones y accesorios de cabello.
- Kits o combos mixtos: según el producto principal del nombre.
- Responde solo con la categoría exacta de la lista.

Formato: una línea por producto, `id,categoría`.
