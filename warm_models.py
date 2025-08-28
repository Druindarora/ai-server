from faster_whisper import WhisperModel

models = ["tiny", "base", "small", "medium"]

for name in models:
    print(f"⏬ Téléchargement du modèle : {name}")
    # CPU + format compatible CPU
    WhisperModel(name, device="cpu", compute_type="int8")
print("✅ Tous les modèles sont téléchargés (cache CPU).")
