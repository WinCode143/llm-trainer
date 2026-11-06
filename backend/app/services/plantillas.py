"""Plantillas de dataset por caso de uso.

La barrera para empezar no suele ser la herramienta: es no saber qué forma tienen que tener los
datos. Cada plantilla trae ejemplos reales que se pueden descargar, editar y volver a subir, y dice
cuántos ejemplos hacen falta y qué método de entrenamiento le corresponde.
"""

PLANTILLAS = {
    "soporte": {
        "nombre": "Atención al cliente",
        "descripcion": "Responder preguntas sobre tu producto con el tono y las normas de tu empresa.",
        "formato": "chat",
        "metodo": "sft",
        "cuantos": "300–1000 ejemplos. Con menos de 200 el modelo copia el tono pero inventa datos.",
        "consejo": "No metas aquí el conocimiento que cambia (precios, stock): eso va en el RAG. El fine-tuning es para el tono y el formato de respuesta.",
        "filas": [
            {"instruction": "¿Cuánto tarda el envío a Canarias?",
             "output": "Los envíos a Canarias tardan entre 5 y 7 días laborables. Ten en cuenta que pueden aplicarse tasas de aduana al recibirlo."},
            {"instruction": "Quiero devolver un pedido que llegó roto",
             "output": "Lamento mucho lo ocurrido. Para gestionarlo necesito el número de pedido y una foto del producto. En cuanto me los envíes tramito la recogida sin coste y el reembolso completo."},
            {"instruction": "esto es una basura, llevo 3 semanas esperando",
             "output": "Entiendo tu enfado y tienes razón en quejarte: tres semanas está muy por encima de nuestro plazo. Dame el número de pedido y lo reviso ahora mismo para darte una fecha concreta o el reembolso, lo que prefieras."},
        ],
    },
    "extraccion": {
        "nombre": "Extraer datos de un texto",
        "descripcion": "Sacar campos estructurados (JSON) de facturas, correos, informes o formularios.",
        "formato": "chat",
        "metodo": "sft",
        "cuantos": "200–500 ejemplos. Es de las tareas que menos datos necesitan porque el patrón es muy regular.",
        "consejo": "Usa siempre exactamente el mismo esquema JSON en todas las respuestas, con las mismas claves y en el mismo orden. Las incoherencias aquí se aprenden.",
        "filas": [
            {"instruction": "Extrae los datos de esta factura en JSON:\nFactura A-2024-0912\nFecha: 12/09/2024\nCliente: Talleres Ruiz S.L.\nBase 1.200,00 € · IVA 21% · Total 1.452,00 €",
             "output": "{\"numero\": \"A-2024-0912\", \"fecha\": \"2024-09-12\", \"cliente\": \"Talleres Ruiz S.L.\", \"base\": 1200.00, \"iva_pct\": 21, \"total\": 1452.00}"},
            {"instruction": "Extrae los datos de esta factura en JSON:\nFAC 77/25 del 3 de enero de 2025, emitida a Marta Gómez, importe 340 euros sin impuestos, IVA reducido del 10%, total 374",
             "output": "{\"numero\": \"77/25\", \"fecha\": \"2025-01-03\", \"cliente\": \"Marta Gómez\", \"base\": 340.00, \"iva_pct\": 10, \"total\": 374.00}"},
        ],
    },
    "clasificacion": {
        "nombre": "Clasificar textos",
        "descripcion": "Asignar una etiqueta de una lista cerrada: temas, urgencia, sentimiento, departamento.",
        "formato": "chat",
        "metodo": "sft",
        "cuantos": "50–100 ejemplos POR CATEGORÍA, y equilibrados: si una categoría tiene diez veces más ejemplos, el modelo la elegirá siempre.",
        "consejo": "Que la respuesta sea sólo la etiqueta, sin explicación. Si quieres el porqué, entrena dos modelos o pide el razonamiento aparte.",
        "filas": [
            {"instruction": "Clasifica la urgencia (baja/media/alta): «No me llega la factura del mes pasado, ¿me la reenviáis cuando podáis?»",
             "output": "baja"},
            {"instruction": "Clasifica la urgencia (baja/media/alta): «La web está caída y no podemos cobrar, tenemos la tienda llena»",
             "output": "alta"},
            {"instruction": "Clasifica la urgencia (baja/media/alta): «El informe de agosto tiene un dato mal, lo necesito para el viernes»",
             "output": "media"},
        ],
    },
    "estilo": {
        "nombre": "Escribir con tu estilo",
        "descripcion": "Que el modelo redacte como tú o como tu marca: vocabulario, longitud, estructura.",
        "formato": "chat",
        "metodo": "sft",
        "cuantos": "100–300 ejemplos, pero de calidad. Aquí importa más lo bueno que lo mucho.",
        "consejo": "Coge textos que ya escribiste y crea la instrucción hacia atrás («escribe un post sobre X»). Es más rápido y el estilo es auténtico.",
        "filas": [
            {"instruction": "Escribe un párrafo anunciando que la tienda cierra por vacaciones del 1 al 15 de agosto.",
             "output": "Cerramos del 1 al 15 de agosto. Los pedidos que entren esos días salen el 16 por orden de llegada, así que si necesitas algo con prisa, mejor pídelo antes del 31. Volvemos con las pilas cargadas."},
        ],
    },
    "preferencias": {
        "nombre": "Preferencias (para DPO)",
        "descripcion": "Enseñar qué respuesta es mejor que otra. Se usa DESPUÉS de un SFT, para afinar tono y forma.",
        "formato": "preference",
        "metodo": "dpo",
        "cuantos": "200–500 pares. No sirve para meter conocimiento nuevo, sólo para elegir entre formas de responder.",
        "consejo": "Que las dos respuestas sean plausibles y se diferencien en UNA cosa (más corta, menos servil, con datos). Si la rechazada es absurda, el modelo no aprende nada útil.",
        "filas": [
            {"prompt": "¿Puedo cancelar mi suscripción?",
             "chosen": "Sí. Entra en Ajustes → Suscripción → Cancelar. Seguirás teniendo acceso hasta el final del periodo ya pagado.",
             "rejected": "¡Claro que sí! 😊 Entiendo perfectamente que quieras cancelar y estaré encantado de ayudarte con este proceso. Cancelar tu suscripción es algo que puedes hacer fácilmente..."},
        ],
    },
    "razonamiento": {
        "nombre": "Razonamiento verificable (para GRPO)",
        "descripcion": "Problemas con una respuesta objetivamente comprobable: cálculos, conversiones, lógica.",
        "formato": "chat",
        "metodo": "grpo",
        "cuantos": "500+ problemas. GRPO necesita volumen porque aprende comparando varios intentos por pregunta.",
        "consejo": "La respuesta del dataset se usa como referencia para puntuar, así que tiene que ser exacta y sin adornos. Si la referencia es ambigua, la recompensa miente.",
        "filas": [
            {"instruction": "Un artículo cuesta 45 € sin IVA. Con el 21% de IVA, ¿cuánto cuesta?", "output": "54.45"},
            {"instruction": "Si 3 operarios tardan 8 horas, ¿cuánto tardan 4 operarios al mismo ritmo?", "output": "6"},
        ],
    },
}


def listado() -> list[dict]:
    return [{"key": k, **{x: v[x] for x in ("nombre", "descripcion", "formato", "metodo", "cuantos", "consejo")},
             "ejemplos": len(v["filas"])} for k, v in PLANTILLAS.items()]
