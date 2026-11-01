# Mana-Agent

[English](README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Brezhoneg](README.br.md) · [Bosanski](README.bs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Français](README.fr.md) · [Ελληνικά](README.gr.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk](README.no.md) · [Polski](README.pl.md) · [Русский](README.ru.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [简体中文](README.zh.md) · [繁體中文](README.zht.md)

Το Mana-Agent είναι μια πλατφόρμα ανοιχτού κώδικα σε Python για τον συντονισμό πρακτόρων τεχνητής νοημοσύνης. Συνδυάζει αυτόνομο βρόχο εκτέλεσης, εξειδικευμένους πράκτορες, μόνιμη διαχείριση εργασιών, εποπτευόμενη χρήση εργαλείων και επεκτάσιμες ενσωματώσεις. Εσείς ορίζετε τον στόχο και το Mana-Agent βοηθά στην επίτευξή του με πραγματικά εργαλεία και έλεγχο αποτελεσμάτων.

## Δυνατότητες

Αναλύει αποθετήρια, έγγραφα και API, σχεδιάζει εργασίες, συντονίζει πράκτορες, παρακολουθεί την πρόοδο, επαναχρησιμοποιεί εγκεκριμένο πλαίσιο και ελέγχει αποτελέσματα. Βοηθά στην έρευνα, τον προγραμματισμό, την αποσφαλμάτωση και την αυτοματοποίηση, καθώς και στη χρήση φυλλομετρητή, Git, SSH, email και Telegram.

**Ζωντανή καθοδήγηση:** Η καθοδήγηση εκτέλεσης σε εξέλιξη, ακόμη και με Ctrl+S, δεν είναι ακόμη διαθέσιμη. Το Plan Mode παραμένει ξεχωριστό και οι απαιτήσεις έγκρισής του εξακολουθούν να ισχύουν. Δείτε τον [σχεδιασμό καθοδήγησης](docs/live-steering-design.md).

Το OpenAI Batch χρησιμοποιείται μόνο κατόπιν ρητού αιτήματος· οι υπόλοιπες ομαδικές εργασίες χρησιμοποιούν εσωτερική παράλληλη εκτέλεση. Οι εργασίες που απαιτούν Batch τίθενται σε μόνιμη αναμονή και συνεχίζονται με τα αποτελέσματα· οι επόμενες συνομιλίες διατηρούν το πλαίσιο χρήστη και βοηθού μεταξύ διεπαφών. [Batch και ανάκτηση](docs/26-openai-batch-api.md).

## Γρήγορη εκκίνηση

Απαιτείται Python 3.10–3.14. Προτεινόμενη εγκατάσταση:

```bash
pipx install git+https://github.com/manaplatform/mana-agent.git
```

Ή από το αποθετήριο:

```bash
git clone https://github.com/manaplatform/mana-agent.git
cd mana-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Ρυθμίστε πάροχο, διαπιστευτήρια και προτιμήσεις μοντέλου με `mana-agent --configure`. Εκτελέστε `mana-agent` στον κατάλογο του έργου. Για τον προαιρετικό πίνακα ελέγχου: `mana-agent dashboard --root-dir .`.

## Άδεια και πληροφορίες

[Άδεια MIT](LICENSE). [Πλήρες αγγλικό README](README.md), [τεκμηρίωση](docs/) και [αναφορά προβλήματος](https://github.com/manaplatform/mana-agent/issues).
