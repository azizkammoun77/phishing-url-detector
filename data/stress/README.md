# Stress-test set

`legit_subdomains.csv` (column `hostname`) is a small, **hand-curated** list of
well-known, legitimate hostnames that have at least one subdomain level
(e.g. `mail.google.com`, `fr.wikipedia.org`, `pytorch.github.io`).

It is used to measure the false-positive rate of the models on a hard case:
the training data tends to associate subdomains with phishing, so legitimate
subdomains are a likely failure mode. Hostnames are normalized like
`src/prepare_data.py` does (lowercase, no leading `www.`) and are only ever
treated as text; nothing here is fetched or resolved.

The list was written by hand and not verified against live sites. It is not
part of the train/test split and must never be used for training.
