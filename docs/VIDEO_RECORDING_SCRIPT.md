# EchoGesture demo video: recording script

**Team Hypervisor · SRMIST.** Final length about **2:55**, which is under the 3-minute limit.

Each person records the clips listed below and sends them to Anirudh. During compiling, the
EchoGesture pipeline runs over the signing footage frame by frame, so **every caption in the final
video is produced by the model, not typed in.** The title, "how it works" and results cards, the HUD,
and the spoken words are all added at compile time. Please **record plain footage**: no
filters, no captions, no edits.

| # | Clip | Who is on camera | Who films | Length |
|---|---|---|---|---|
| 1 | `01_anirudh_intro.mp4` | Anirudh (talking) | Kiran | ~30 s |
| 2 | `02_kiran_sign_A.mp4`, `_B`, `_C` | Kiran (signing) | Koushigan | 3 × ~12 s |
| 3 | `03_kiran_line.mp4` | Kiran (talking) | Koushigan | ~10 s |
| 4 | `04_koushigan_how.mp4` | Koushigan (talking) | Anirudh | ~40 s |
| 5 | `05_kiran_stress.mp4` | Kiran (signing, dim light) | Koushigan | ~12 s |
| 6 | `06_anirudh_close.mp4` | Anirudh (talking) | Kiran | ~25 s |

---

## Camera settings (everyone)

- Phone **horizontal (landscape)**, **1080p at 30 fps**. Not 60 fps, not 4K, not slow-mo.
- Turn off beauty filters, portrait/bokeh mode and stabilisation "action" modes.
- Hold the phone steady, or prop it on a stack of books. Don't zoom while recording.
- Start recording **2 seconds before** speaking or signing, and stop **2 seconds after**.
- Record **at least 2 takes** of every clip. Send all of them; the best take will be picked.

### Talking clips (1, 3, 4, 6)
- Quiet room, with the fan and AC off if possible. Phone about 1 m away, at eye level.
- Face a window or a lamp so the light is on your face, not behind you.
- Frame from mid-chest up. Speak a little slower than normal. Reading from a laptop just behind the
  phone is fine.

### Signing clips (2, 5): important
These shots are **the view through the smart glasses**, so the camera stands where the person
talking to Kiran would stand.
- Koushigan holds the phone at **his own eye level**, **1.5 to 2 m** from Kiran.
- Frame **from the top of Kiran's head to his waist**. **Both hands must stay in frame** for the whole
  sign, including when they move out to the sides.
- Use a plain wall behind Kiran and light his face and hands from the front. Wear a **plain, dark,
  long-sleeved top**, which makes the hands easy to track. No rings or watches.
- **Take A:** face the camera directly. **Take B:** Kiran turns about **20–30°** to one side, which
  shows the system handles angled signers. **Take C:** either angle.
- **Rhythm:** start with hands resting at the waist (1 s) → sign the word → **drop the hands to the
  waist and pause for 1 s** → next word. The pause is how the system knows one word has ended.

---

## The signs (Kiran)

The model knows 100 ASL words. These phrases use words it recognises most reliably on new signers.
Learn each sign from a video dictionary, ideally **Lifeprint (ASL University)**, **SigningSavvy** or
**Handspeak**, which match the style the model was trained on. Copy the **dominant-hand** version
exactly: handshape, location, movement, and the number of repetitions.

| Take | ASL words, in order | What the HUD will caption |
|---|---|---|
| A (facing camera) | **MOTHER** · **WORK** · **FINISH** | "Mother work finish": Mom finished work |
| B (turned 20–30°) | **MAN** · **LIKE** · **BASKETBALL** | "Man like basketball" |
| C (either) | **WOMAN** · **DRINK** · **HOT** | "Woman drink hot" |
| Stress (clip 5) | **HELP** · **NO** | "Help no" |

**Backup words** if one of the above doesn't get recognised: TABLE, CHAIR, BED, COMPUTER, FISH,
CLOTHES, ORANGE, WHITE, TALL, DARK, PLAY, KISS, HAT, BIRD.

Tip: practise each phrase about 10 times before recording, so it's smooth and at a natural speed.
Sign at a natural pace, not slowed down.

**Clip 5 (stress test):** same framing, but **turn the main light off** so the room is dim (still
visible, not black). Partway through, Koushigan briefly holds a book or notebook so it covers
**part** of the frame (not Kiran's hands for the whole sign).

---

## Spoken lines

### Clip 1 · Anirudh · intro (~30 s)
> Hi, we're Team Hypervisor from SRMIST. I'm Anirudh, and with me are Kiran and Koushigan.
> This is EchoGesture, our project for Track 2.
>
> Over seventy million deaf people use sign language, but most people they talk to can't sign.
> Smart glasses could caption that conversation in real time, but only if the translation keeps up
> with natural signing, and only if video of the people around you never leaves the device.
>
> So that's what we built. Here's what the glasses see.

### Clips 2A–2C · Kiran · signing (no talking)
*(Silent. The captions and spoken words are added during compiling.)*

### Clip 3 · Kiran · after signing (~10 s)
> That was me signing, filmed from where a glasses wearer would stand.
> Every caption you just saw came from the model, frame by frame. Nothing was typed in by hand.

### Clip 4 · Koushigan · how it works (~40 s)
> Here's how it works. The camera tracks fifty-five points on the signer's hands and upper body,
> right on the device.
> A tiny model, three small networks compressed to eight-bit and under six megabytes in total,
> reads that motion and predicts the word in under two milliseconds.
>
> The caption floats right beside the signer, never over their face or hands, because in sign
> language those carry the meaning. And if the model isn't sure, it shows that in amber instead of
> pretending.
>
> Not a single frame of video is uploaded. Let's make it harder: low light, and something in the way.

### Clip 5 · Kiran · stress test
*(Silent. Sign HELP · NO in dim light with the partial occluder.)*

### Clip 6 · Anirudh · results and close (~25 s)
> On signers it has never seen before, EchoGesture recognises a hundred ASL words with seventy-one
> percent top-1 and eighty-nine percent top-5 accuracy, at under forty milliseconds per frame on a
> single CPU, with every model on the device under twenty megabytes.
>
> Our code, model and benchmarks are open on GitHub.
> EchoGesture: conversation without the gap, privacy without compromise. Thank you.

---

## Running order of the final video (compiled)

| Time | Content |
|---|---|
| 0:00–0:06 | Title card: EchoGesture · Team Hypervisor · SRMIST |
| 0:06–0:36 | Clip 1, Anirudh intro |
| 0:36–1:16 | Clips 2A–2C, Kiran signing through the glasses HUD (live captions + voice) |
| 1:16–1:26 | Clip 3, Kiran's line |
| 1:26–2:06 | Clip 4, Koushigan, with the "how it works" diagram overlaid |
| 2:06–2:20 | Clip 5, stress test through the HUD |
| 2:20–2:48 | Clip 6, Anirudh, with the results card overlaid |
| 2:48–2:55 | End card: GitHub link |

## Sending the files
Put every take in one Google Drive folder (or straight into `echogesture/demo/raw/` on Anirudh's
laptop) **with the file names above**, adding `_take1`, `_take2`, … for example
`02_kiran_sign_A_take2.mp4`. Don't trim or compress anything. Send the originals.
