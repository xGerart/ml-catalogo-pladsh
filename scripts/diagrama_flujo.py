# Prompt (IA): «Dibuja en SVG, con el estilo de los diagramas de mi guía de estudio, el flujo
# completo del proyecto: desde el export del ERP hasta el catálogo categorizado, con todo lo que usa.»
"""Genera reports/figures/diagrama_flujo.svg con el estilo de la guía de estudio.

Uso: python scripts/diagrama_flujo.py reports/figures/diagrama_flujo.svg
El PNG que muestra el notebook se exporta de este SVG con un navegador (Colab no
siempre renderiza salidas SVG)."""
import sys

C = {  # fill, stroke (paleta de guia_img/)
    "datos": ("#e9ebef", "#5b6573"),
    "explora": ("#dff3ef", "#0f7b6c"),
    "etiq": ("#f8ecdb", "#a55a06"),
    "train": ("#e5e8fb", "#3446c2"),
    "t2": ("#ede6f8", "#6e45b0"),
    "test": ("#f8e3eb", "#b02d62"),
    "ok": ("#e2f2e4", "#2e7d32"),
}
TXT = "#3d4450"
out = []
add = out.append


def box(x, y, w, h, kind, title, *lines):
    f, s = C[kind]
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" stroke-width="1.6" style="fill:{f};stroke:{s}"/>')
    cx = x + w / 2
    n = 1 + len(lines)
    y0 = y + h / 2 - (n - 1) * 8 + 4
    add(f'<text x="{cx}" y="{y0}" text-anchor="middle" font-size="12.5" font-weight="700" style="fill:{s}">{title}</text>')
    for i, ln in enumerate(lines, 1):
        add(f'<text x="{cx}" y="{y0 + 16 * i}" text-anchor="middle" font-size="11.5" style="fill:{TXT}">{ln}</text>')


def arrow(pts, label=None, lx=None, ly=None, color="#6b7280", marker="a", dash=False, anchor="start"):
    d = " ".join(f"{x},{y}" for x, y in pts)
    extra = ' stroke-dasharray="5 4"' if dash else ""
    add(f'<polyline points="{d}" fill="none" stroke-width="1.4" style="stroke:{color}"{extra} marker-end="url(#{marker})"/>')
    if label:
        add(f'<text x="{lx}" y="{ly}" text-anchor="{anchor}" font-size="11" style="fill:{color}">{label}</text>')


W, H = 1000, 1130
# fila 1-2: datos
box(350, 20, 260, 56, "datos", "Export del ERP", "11 296 productos · id y nombre")
arrow([(480, 76), (480, 108)])
box(350, 110, 260, 56, "datos", "Limpieza y deduplicado", "10 777 nombres únicos")
# fila 3: tres destinos
box(60, 210, 240, 56, "explora", "Clustering exploratorio", "k-means → taxonomía de 8 clases")
box(360, 210, 240, 56, "etiq", "Muestra aleatoria", "1 000 productos (semilla 42)")
box(850, 210, 130, 56, "datos", "Resto", "9 777 sin etiqueta")
arrow([(400, 166), (400, 186), (180, 186), (180, 208)], "todo el catálogo", 190, 181)
arrow([(480, 166), (480, 208)], "al azar", 488, 194)
arrow([(560, 166), (560, 186), (915, 186), (915, 208)], "el resto", 860, 181)
# fila 4: etiquetado
box(300, 310, 360, 56, "etiq", "Etiquetado asistido", "un LLM propone · una persona revisa las 1 000")
arrow([(180, 266), (180, 338), (298, 338)], "taxonomía", 188, 330)
arrow([(480, 266), (480, 308)])
# fila 5: split
box(270, 420, 230, 56, "train", "Entrenamiento", "800 productos (80 %)")
box(600, 420, 220, 56, "test", "Test", "200 productos · apartado")
arrow([(420, 366), (420, 392), (385, 392), (385, 418)], "split estratificado", 428, 388)
arrow([(540, 366), (540, 392), (710, 392), (710, 418)], color="#b02d62", marker="m")
# grupo optimización
add('<rect x="40" y="505" width="580" height="160" rx="14" fill="none" stroke-width="1.6" stroke-dasharray="6 5" style="stroke:#3446c2"/>')
add('<text x="330" y="523" text-anchor="middle" font-size="11.5" font-weight="700" letter-spacing=".04em" style="fill:#3446c2">OPTIMIZACIÓN · GridSearchCV 5 × 3</text>')
box(60, 530, 260, 120, "train", "Técnica 1", "TF-IDF de n-gramas", "de caracteres", "→ χ² (selección de k)", "→ regresión logística (C = 1/λ)")
box(340, 530, 260, 120, "t2", "Técnica 2", "Word2Vec preentrenado", "(SBWCE, 300 dimensiones)", "→ promedio → escalado", "→ SVM kernel RBF (C, γ)")
arrow([(385, 476), (385, 492), (190, 492), (190, 528)], "nombres y etiquetas", 393, 488)
arrow([(385, 476), (385, 492), (470, 492), (470, 528)])
# comparación
box(60, 715, 540, 56, "ok", "Comparación estadística", "repeated stratified 10 × 10 (mismos folds) → Wilcoxon + t corregido")
arrow([(190, 650), (190, 713)], "mejor configuración", 198, 688)
arrow([(470, 650), (470, 713)], "mejor configuración", 478, 688)
# evaluación en test
box(560, 815, 300, 56, "test", "Evaluación en test", "métricas · ROC · matriz de confusión")
arrow([(710, 476), (710, 813)], "se usa solo al final", 718, 790, color="#b02d62", marker="m")
arrow([(600, 743), (660, 743), (660, 813)], "ganadora", 604, 737)
# modelo final
box(60, 815, 440, 56, "ok", "Modelo final", "ganadora reentrenada con los 1 000 · umbral 0,8")
arrow([(330, 771), (330, 813)], "técnica ganadora", 338, 797)
# aplicación
box(250, 925, 500, 56, "datos", "Aplicación al catálogo restante", "cada producto → categoría predicha + confianza")
arrow([(280, 871), (280, 923)], "predice", 288, 902)
arrow([(915, 266), (915, 953), (752, 953)], "9 777 nombres", 908, 945, anchor="end")
# salidas
box(150, 1040, 300, 56, "ok", "Categorizado automático", "confianza ≥ 0,8")
box(550, 1040, 300, 56, "datos", "Revisión manual", "confianza &lt; 0,8 · nombres ambiguos")
arrow([(400, 981), (400, 1010), (300, 1010), (300, 1038)])
arrow([(600, 981), (600, 1010), (700, 1010), (700, 1038)])

claim = ("Flujo del proyecto: los nombres del ERP se deduplican; una muestra de 1 000 se etiqueta "
         "con asistencia y entrena dos técnicas que se optimizan y comparan estadísticamente; la ganadora, "
         "evaluada en un test apartado, categoriza el resto del catálogo y deja los casos dudosos para revisión manual.")
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{claim}" font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif">',
       '<defs>',
       '<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:#6b7280"/></marker>',
       '<marker id="m" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:#b02d62"/></marker>',
       '</defs>',
       f'<rect width="{W}" height="{H}" style="fill:#f5f6f8"/>'] + out + ["</svg>"]
open(sys.argv[1], "w").write("\n".join(svg) + "\n")
