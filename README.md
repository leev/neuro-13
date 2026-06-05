# Neuro 13

Neuro 13 is a kid-friendly, hands-on introduction to neural networks. It uses a single-page HTML lesson plus small Python exercises to explain neurons, weights, bias, linearly separable problems, hidden layers, XOR, and related beginner concepts.

The source page is `neuro-13.html`. Running `make pages` builds the deployable static site in `public/` by copying the HTML to `public/index.html` and including the exercise Python files.

The site deploys to:

https://neuro-13.leev.net

GitHub Actions builds `public/` and deploys it to Cloudflare Workers using the configuration in `wrangler.toml`.
