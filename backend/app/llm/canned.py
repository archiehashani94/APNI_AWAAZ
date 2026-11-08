"""Offline fallback replies.

These are used when no LLM is configured or the LLM call fails. They are still
intent-aware, so a question about water never gets the same three lines as a
question about pain. Keep this file dependency-free and fast.
"""

from __future__ import annotations

import re

from app.llm.interface import ReplyOption

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
HINGLISH = re.compile(
    r"\b(aap|theek|kya|hai|ho|hain|chahiye|haan|nahi|nahin|paani|pyaas|"
    r"dhanyavaad|shukriya|namaste|thoda|bilkul|khaana|khana|bhookh|dard|"
    r"takleef|madad|dawa|dawai|shauchalay|washroom|alvida)\b",
    re.I,
)
QUESTION_WORDS = (
    "?",
    "kya",
    "kaise",
    "kab",
    "kahan",
    "kaun",
    "kitna",
    "do you",
    "did you",
    "are you",
    "have you",
    "can you",
    "will you",
    "what",
    "when",
    "where",
    "who",
    "how",
)

# intent -> (keywords, emotions, replies)
# "hi" is Devanagari and "hinglish" is Latin-script Hindi.
INTENTS: dict[str, tuple[tuple[str, ...], tuple[str, ...], dict[str, tuple[str, ...]]]] = {
    "pain": (
        ("pain", "hurt", "ache", "dard", "takleef", "दर्द", "तकलीफ"),
        ("urgent", "warm", "urgent"),
        {
            "en": ("Yes, I'm in pain.", "It hurts a little.", "I need help with the pain."),
            "hi": ("हाँ, मुझे दर्द हो रहा है।", "थोड़ा दर्द है।", "दर्द में मदद चाहिए।"),
            "hinglish": (
                "Haan, mujhe dard ho raha hai.",
                "Thoda dard hai.",
                "Dard mein madad chahiye.",
            ),
        },
    ),
    "water": (
        ("water", "paani", "pyaas", "thirst", "drink", "पानी", "प्यास"),
        ("warm", "calm", "warm"),
        {
            "en": ("Yes, water please.", "No, thank you.", "A little water, please."),
            "hi": ("हाँ, पानी दीजिए।", "नहीं, धन्यवाद।", "थोड़ा पानी दीजिए।"),
            "hinglish": ("Haan, paani dijiye.", "Nahi, dhanyavaad.", "Thoda paani dijiye."),
        },
    ),
    "food": (
        ("food", "khana", "khaana", "bhookh", "hungry", "eat", "meal", "खाना", "भूख"),
        ("warm", "calm", "calm"),
        {
            "en": ("Yes, I'm hungry.", "No, I've eaten.", "Maybe a little later."),
            "hi": ("हाँ, मुझे भूख लगी है।", "नहीं, मैं खा चुका हूँ।", "शायद थोड़ी देर बाद।"),
            "hinglish": (
                "Haan, mujhe bhookh lagi hai.",
                "Nahi, main kha chuka hoon.",
                "Shayad thodi der baad.",
            ),
        },
    ),
    "help": (
        ("help", "madad", "assist", "मदद"),
        ("urgent", "warm", "calm"),
        {
            "en": ("Yes, please help me.", "I need some help.", "I can manage for now."),
            "hi": ("हाँ, कृपया मेरी मदद कीजिए।", "मुझे थोड़ी मदद चाहिए।", "अभी मैं संभल सकता हूँ।"),
            "hinglish": (
                "Haan, please meri madad kijiye.",
                "Mujhe thodi madad chahiye.",
                "Abhi main sambhal sakta hoon.",
            ),
        },
    ),
    "toilet": (
        ("toilet", "bathroom", "washroom", "shauchalay", "loo", "शौचालय"),
        ("urgent", "calm", "urgent"),
        {
            "en": ("I need the washroom.", "Not right now.", "Please help me to the washroom."),
            "hi": ("मुझे शौचालय जाना है।", "अभी नहीं।", "कृपया शौचालय तक मदद कीजिए।"),
            "hinglish": (
                "Mujhe washroom jaana hai.",
                "Abhi nahi.",
                "Please washroom tak madad kijiye.",
            ),
        },
    ),
    "medicine": (
        ("medicine", "dawa", "dawai", "tablet", "pill", "दवा"),
        ("warm", "calm", "calm"),
        {
            "en": ("Yes, I need my medicine.", "Not yet, thank you.", "I've already taken it."),
            "hi": ("हाँ, मुझे दवा चाहिए।", "अभी नहीं, धन्यवाद।", "मैं दवा ले चुका हूँ।"),
            "hinglish": (
                "Haan, mujhe dawa chahiye.",
                "Abhi nahi, dhanyavaad.",
                "Main dawa le chuka hoon.",
            ),
        },
    ),
    "greeting": (
        ("hello", "hi", "hey", "namaste", "namaskar", "नमस्ते"),
        ("warm", "warm", "warm"),
        {
            "en": ("Hello.", "Hi, thank you for coming.", "Hello, I'm glad you're here."),
            "hi": ("नमस्ते।", "नमस्ते, आने के लिए धन्यवाद।", "नमस्ते, आपको देखकर अच्छा लगा।"),
            "hinglish": (
                "Namaste.",
                "Namaste, aane ke liye dhanyavaad.",
                "Namaste, aapko dekhkar achha laga.",
            ),
        },
    ),
    "howareyou": (
        ("how are you", "kaise ho", "kaise hain", "theek ho", "theek hain", "कैसे हो", "कैसे हैं"),
        ("warm", "calm", "calm"),
        {
            "en": ("I'm okay, thank you.", "Not great today.", "I'm managing, thank you."),
            "hi": ("मैं ठीक हूँ, धन्यवाद।", "आज थोड़ा ठीक नहीं हूँ।", "मैं संभल रहा हूँ, धन्यवाद।"),
            "hinglish": (
                "Main theek hoon, dhanyavaad.",
                "Aaj thoda theek nahi hoon.",
                "Main sambhal raha hoon, dhanyavaad.",
            ),
        },
    ),
    "thanks": (
        ("thank", "thanks", "dhanyavaad", "shukriya", "धन्यवाद", "शुक्रिया"),
        ("warm", "warm", "warm"),
        {
            "en": ("You're welcome.", "No need to thank me.", "I'm happy to help."),
            "hi": ("आपका स्वागत है।", "धन्यवाद कहने की ज़रूरत नहीं।", "मदद करके खुशी हुई।"),
            "hinglish": (
                "Aapka swagat hai.",
                "Dhanyavaad kehne ki zaroorat nahi.",
                "Madad karke khushi hui.",
            ),
        },
    ),
    "bye": (
        ("bye", "goodbye", "good night", "goodnight", "alvida", "अलविदा"),
        ("warm", "warm", "warm"),
        {
            "en": ("Goodbye.", "See you soon.", "Take care."),
            "hi": ("अलविदा।", "फिर मिलेंगे।", "अपना ध्यान रखिए।"),
            "hinglish": ("Alvida.", "Phir milenge.", "Apna dhyan rakhiye."),
        },
    ),
}

YES_NO = {
    "en": ("Yes.", "No.", "Please repeat that."),
    "hi": ("हाँ।", "नहीं।", "कृपया दोबारा कहिए।"),
    "hinglish": ("Haan.", "Nahi.", "Please dobara kahiye."),
}
DEFAULT = {
    "en": ("I understand.", "Can you say that again?", "Give me a moment, please."),
    "hi": ("मैं समझ गया।", "क्या आप दोबारा कह सकते हैं?", "कृपया मुझे एक पल दीजिए।"),
    "hinglish": ("Main samajh gaya.", "Kya aap dobara keh sakte hain?", "Please mujhe ek pal dijiye."),
}
DEFAULT_EMOTIONS = ("warm", "calm", "calm")


def _language(text: str) -> str:
    if DEVANAGARI.search(text):
        return "hi"
    if HINGLISH.search(text):
        return "hinglish"
    return "en"


def _matches(text: str, keyword: str) -> bool:
    if keyword in {"?", "दर्द", "तकलीफ", "पानी", "प्यास", "खाना", "भूख", "मदद", "शौचालय", "दवा", "नमस्ते", "कैसे हो", "कैसे हैं", "धन्यवाद", "शुक्रिया", "अलविदा"}:
        return keyword in text
    return re.search(rf"\b{re.escape(keyword)}\b", text) is not None


def canned_replies(turns: list[dict[str, str]]) -> list[ReplyOption]:
    last = next((t.get("text", "") for t in reversed(turns) if t.get("role") == "partner"), "")
    last = last.strip()
    language = _language(last)
    lowered = last.lower()

    if last:
        for intent, (keywords, emotions, replies) in INTENTS.items():
            if any(_matches(lowered, keyword) for keyword in keywords):
                return [
                    ReplyOption(text, emotion)
                    for text, emotion in zip(replies[language], emotions)
                ]

    if last and any(_matches(lowered, word) for word in QUESTION_WORDS):
        return [
            ReplyOption(text, emotion)
            for text, emotion in zip(YES_NO[language], DEFAULT_EMOTIONS)
        ]

    return [
        ReplyOption(text, emotion)
        for text, emotion in zip(DEFAULT[language], DEFAULT_EMOTIONS)
    ]
