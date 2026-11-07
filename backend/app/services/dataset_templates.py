"""Dataset templates by use case.

What stops people from starting usually isn't the tool: it's not knowing what shape the data needs
to have. Each template comes with real examples you can download, edit and upload again, and says
how many examples you need and which training method fits.
"""

TEMPLATES = {
    "support": {
        "nombre": "Customer support",
        "descripcion": "Answer questions about your product with your company's tone and policies.",
        "formato": "chat",
        "metodo": "sft",
        "cuantos": "300–1000 examples. With fewer than 200 the model copies the tone but makes up facts.",
        "consejo": "Don't put knowledge that changes (prices, stock) here: that belongs in RAG. Fine-tuning is for the tone and the answer format.",
        "filas": [
            {"instruction": "How long does shipping to Hawaii take?",
             "output": "Shipping to Hawaii takes 5 to 7 business days. Keep in mind that some carriers add a remote-area surcharge."},
            {"instruction": "I want to return an order that arrived broken",
             "output": "I'm really sorry about that. To sort it out I need your order number and a photo of the product. As soon as you send them, I'll arrange a free pickup and a full refund."},
            {"instruction": "this is garbage, I've been waiting 3 weeks",
             "output": "I understand you're upset, and you're right to complain: three weeks is far beyond our delivery window. Give me your order number and I'll look into it right now to get you a firm date or a refund, whichever you prefer."},
        ],
    },
    "extraction": {
        "nombre": "Extract data from text",
        "descripcion": "Pull structured fields (JSON) out of invoices, emails, reports or forms.",
        "formato": "chat",
        "metodo": "sft",
        "cuantos": "200–500 examples. One of the tasks that needs the least data, because the pattern is very regular.",
        "consejo": "Always use exactly the same JSON schema in every answer, with the same keys in the same order. Inconsistencies here get learned.",
        "filas": [
            {"instruction": "Extract this invoice's data as JSON:\nInvoice A-2024-0912\nDate: 09/12/2024\nCustomer: Ruiz Auto Repair LLC\nSubtotal $1,200.00 · Tax 8% · Total $1,296.00",
             "output": "{\"number\": \"A-2024-0912\", \"date\": \"2024-09-12\", \"customer\": \"Ruiz Auto Repair LLC\", \"subtotal\": 1200.00, \"tax_pct\": 8, \"total\": 1296.00}"},
            {"instruction": "Extract this invoice's data as JSON:\nINV 77/25 dated January 3, 2025, billed to Martha Gomez, $340 before tax, 5% tax, total 357",
             "output": "{\"number\": \"77/25\", \"date\": \"2025-01-03\", \"customer\": \"Martha Gomez\", \"subtotal\": 340.00, \"tax_pct\": 5, \"total\": 357.00}"},
        ],
    },
    "classification": {
        "nombre": "Classify texts",
        "descripcion": "Assign a label from a closed list: topics, urgency, sentiment, department.",
        "formato": "chat",
        "metodo": "sft",
        "cuantos": "50–100 examples PER CATEGORY, and balanced: if one category has ten times more examples, the model will always pick it.",
        "consejo": "Make the answer just the label, with no explanation. If you want the reasoning, train two models or ask for it separately.",
        "filas": [
            {"instruction": "Classify the urgency (low/medium/high): “I didn't get last month's invoice, could you resend it when you can?”",
             "output": "low"},
            {"instruction": "Classify the urgency (low/medium/high): “The website is down and we can't take payments, the store is full”",
             "output": "high"},
            {"instruction": "Classify the urgency (low/medium/high): “The August report has a wrong figure, I need it by Friday”",
             "output": "medium"},
        ],
    },
    "style": {
        "nombre": "Write in your style",
        "descripcion": "Have the model write like you or your brand: vocabulary, length, structure.",
        "formato": "chat",
        "metodo": "sft",
        "cuantos": "100–300 examples, but good ones. Here quality matters more than quantity.",
        "consejo": "Take texts you've already written and work the instruction out backwards (“write a post about X”). It's faster and the style is genuine.",
        "filas": [
            {"instruction": "Write a paragraph announcing that the store is closed for vacation from August 1 to 15.",
             "output": "We're closed August 1–15. Orders placed during those days ship on the 16th in the order they came in, so if you need something soon, order before July 31. We'll be back with our batteries recharged."},
        ],
    },
    "preferences": {
        "nombre": "Preferences (for DPO)",
        "descripcion": "Teach which answer is better than another. Used AFTER an SFT, to fine-tune tone and form.",
        "formato": "preference",
        "metodo": "dpo",
        "cuantos": "200–500 pairs. It can't add new knowledge, only choose between ways of answering.",
        "consejo": "Make both answers plausible and different in ONE thing (shorter, less servile, with data). If the rejected one is absurd, the model learns nothing useful.",
        "filas": [
            {"prompt": "Can I cancel my subscription?",
             "chosen": "Yes. Go to Settings → Subscription → Cancel. You'll keep access until the end of the period you've already paid for.",
             "rejected": "Of course you can! 😊 I completely understand that you'd like to cancel and I'd be delighted to help you with this process. Cancelling your subscription is something you can easily do..."},
        ],
    },
    "reasoning": {
        "nombre": "Verifiable reasoning (for GRPO)",
        "descripcion": "Problems with an objectively checkable answer: calculations, conversions, logic.",
        "formato": "chat",
        "metodo": "grpo",
        "cuantos": "500+ problems. GRPO needs volume because it learns by comparing several attempts per question.",
        "consejo": "The dataset's answer is used as the reference for scoring, so it has to be exact and unadorned. If the reference is ambiguous, the reward lies.",
        "filas": [
            {"instruction": "An item costs $45 before tax. With 8% sales tax, how much does it cost?", "output": "48.6"},
            {"instruction": "If 3 workers take 8 hours, how long do 4 workers take at the same pace?", "output": "6"},
        ],
    },
}


def listing() -> list[dict]:
    return [{"key": k, **{x: v[x] for x in ("nombre", "descripcion", "formato", "metodo", "cuantos", "consejo")},
             "ejemplos": len(v["filas"])} for k, v in TEMPLATES.items()]
