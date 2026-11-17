# Neuro 13

Neuro 13 is a kid-friendly, hands-on introduction to neural networks. It uses a single-page HTML lesson plus small Python exercises to explain neurons, weights, bias, linearly separable problems, hidden layers, XOR, and related beginner concepts.

The source page is `neuro-13.html`. Running `make pages` builds the deployable static site in `public/` by copying the HTML to `public/index.html` and including the exercise Python files.

`headlines.csv` holds 10,000 labelled news headlines (a cleaned, balanced sample of the AG News research collection) used by the decision-model chapter and `neuro10.py`. Rebuild it with `tools/make_headlines.py`. The chapter's widget and in-browser Python fetch the file, so preview the page through a local server (`python3 -m http.server`) rather than opening it as a file.

The site deploys to:

https://neuro-13.leev.net

GitHub Actions builds `public/` and deploys it to Cloudflare Workers using the configuration in `wrangler.toml`.
