---
title: Fairview
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Fair face verification that asks a person when unsure
---

# Fairview

Face verification that knows when to say "not sure". Every face gets one of
four outcomes: match confirmed, needs review (sent to a guard), no match (a
visitor, not a suspect) or possible spoof. Scores are calibrated so error rates
even out across skin-type groups without ever reading race or skin tone.

Built by CTRL ALT Elite for the Lenovo problem statement at the NC A&T
hackathon. Source: https://github.com/fukubie/fairview

## Using this Space

Open the app at its own address (the "Embed this Space" menu gives it); the
camera works best there. No sign-in is needed. This is a short-lived demo with
no persistent storage: enrolled faces, the activity log and review photos are
wiped whenever the Space restarts, and the Space is taken down after the demo.

The face models are InsightFace and AdaFace weights released for
non-commercial research use only. This Space is a non-commercial demo.
