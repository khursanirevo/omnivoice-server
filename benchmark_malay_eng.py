"""Generate audio for 50 mixed English/Malay sentences with MPS benchmarking."""
import asyncio
import io
import os
import time
import wave

import aiohttp

OUTPUT_DIR = "output_malay_eng"
REF_AUDIO_PATH = "anwar.wav"

MIXED_SENTENCES = [
    "Selamat pagi, hari ini cuaca sangat cerah sekali.",
    "Saya pergi ke kedai untuk beli some groceries.",
    "The meeting will start pukul dua petang nanti.",
    "Boleh tolong passkan remote control tu?",
    "My friend akan datang ke rumah esok pagi.",
    "Kereta saya rosak so I need to send it to the workshop.",
    "Dia sangat pandai dalam mathematics dan science.",
    "We should go makan at that new restaurant dekat Bangsar.",
    "Anak saya suka watch cartoons pada waktu petang.",
    "The weather hari ni sangat panas, kan?",
    "Saya dah finish semua homework untuk minggu ni.",
    "Please jangan lupa bawa laptop kamu esok.",
    "Kucing tu sangat cute, I want to adopt one too.",
    "My boss kata project ni perlu siap before Friday.",
    "Jalan jam sangat hari ni, took me one hour to reach office.",
    "Dia punya English sangat fluent sebab dia pergi study abroad.",
    "Can you tolong translate ayat ni ke Bahasa Melayu?",
    "Mak saya masak chicken curry yang sangat sedap semalam.",
    "The kids sedang bermain di taman belakang rumah.",
    "Saya plan nak pergi travel ke Japan next year.",
    "The exam result akan keluar dekat hujung bulan ni.",
    "Abang saya kerja sebagai engineer dekat company besar.",
    "I think kita patut start saving money untuk masa depan.",
    "Harga barang semakin mahal nowadays, susah nak survive.",
    "She is one of the best students dalam kampus kami.",
    "Kami pergi picnic last weekend dekat Pantai Port Dickson.",
    "The internet connection kat sini sangat slow hari ni.",
    "Saya baru je belajar how to cook nasi lemak from scratch.",
    "They cancelled the event sebab hujan lebat sangat.",
    "My neighbour sangat friendly, selalu bagi kami kuih raya.",
    "Kamu dah prepare untuk presentation esok atau belum?",
    "The doctor advise saya untuk exercise lebih kerap.",
    "Buku ni sangat interesting, I cannot put it down.",
    "Dia kata dia akan call back lepas lunch break nanti.",
    "We need to discuss about the budget untuk project ni.",
    "Saya suka dengar muzik while driving balik dari kerja.",
    "The teacher explained the formula with sangat jelas sekali.",
    "Mereka akan buat open house sempena Hari Raya Aidilfitri.",
    "I accidentally deleted the file, ada cara untuk recover tak?",
    "Kedai kopi tu baru buka branch dekat area kami.",
    "The flight was delayed selama tiga jam semalam.",
    "Saya rasa this idea boleh work kalau kita execute properly.",
    "Dia punya motivation sangat tinggi to achieve his dreams.",
    "Please hantar report tu sebelum pukul lima petang ni.",
    "The supermarket dekat sini ada sale besar this weekend.",
    "Saya dah book hotel untuk our trip ke Pulau Langkawi.",
    "She recommended satu buku tentang self improvement dan mindset.",
    "Kita kena submit application before the deadline on March.",
    "The sunset view dari atas bukit tu memang breathtaking sangat.",
    "Saya harap everything goes well untuk interview esok pagi.",
]


async def generate_single(session, url, text, idx, output_dir, ref_audio_bytes):
    data_payload = aiohttp.FormData()
    data_payload.add_field("text", text)
    data_payload.add_field(
        "ref_audio", ref_audio_bytes, filename="ref.wav", content_type="audio/wav"
    )
    start = time.perf_counter()
    async with session.post(f"{url}/v1/audio/speech/clone", data=data_payload) as resp:
        data = await resp.read()
    wall = time.perf_counter() - start

    duration = 0
    if len(data) > 44 and data[:4] == b"RIFF":
        with wave.open(io.BytesIO(data), "rb") as wf:
            duration = wf.getnframes() / wf.getframerate()

    # Save audio file
    if resp.status == 200:
        fname = f"{idx:03d}.wav"
        fpath = os.path.join(output_dir, fname)
        with open(fpath, "wb") as f:
            f.write(data)
    payload = {
        "model": "omnivoice",
        "input": text,
        "voice": "alloy",
        "response_format": "wav",
    }
    start = time.perf_counter()
    async with session.post(f"{url}/v1/audio/speech", json=payload) as resp:
        data = await resp.read()
    wall = time.perf_counter() - start

    duration = 0
    if len(data) > 44 and data[:4] == b"RIFF":
        with wave.open(io.BytesIO(data), "rb") as wf:
            duration = wf.getnframes() / wf.getframerate()

    return {
        "idx": idx,
        "status": resp.status,
        "wall_s": round(wall, 3),
        "audio_duration_s": round(duration, 3),
        "rtf": round(wall / duration, 3) if duration > 0 else float("inf"),
        "text_len": len(text),
        "audio_bytes": len(data),
    }


async def main():
    url = "http://localhost:8880"
    n = len(MIXED_SENTENCES)
    print(f"Generating audio for {n} mixed English/Malay sentences")
    print(f"Server: {url}")
    print("Concurrency: 4 workers, sending in batches of 4")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Audio files saved to: {OUTPUT_DIR}/")

    # Load reference audio once
    with open(REF_AUDIO_PATH, "rb") as f:
        ref_audio_bytes = f.read()
    print(f"Reference audio: {REF_AUDIO_PATH} ({len(ref_audio_bytes)} bytes)")

    # Warmup: 5 clone requests to prime CUDA kernels
    warmup_count = 5
    print(f"\n--- Warmup ({warmup_count} clone requests) ---")
    warmup_text = "Hello world, this is a warmup request."
    async with aiohttp.ClientSession() as session:
        for i in range(warmup_count):
            form = aiohttp.FormData()
            form.add_field("text", warmup_text)
            form.add_field(
                "ref_audio", ref_audio_bytes,
                filename="ref.wav", content_type="audio/wav",
            )
            t0 = time.perf_counter()
            async with session.post(f"{url}/v1/audio/speech/clone", data=form) as resp:
                await resp.read()
            print(f"  Warmup {i+1}/{warmup_count}: {time.perf_counter()-t0:.2f}s")
    print("Warmup complete.")

    print("\n--- Sequential Baseline (first 10) ---")
    seq_results = []
    async with aiohttp.ClientSession() as session:
        t0 = time.perf_counter()
        for i, text in enumerate(MIXED_SENTENCES[:10]):
            r = await generate_single(session, url, text, i, OUTPUT_DIR, ref_audio_bytes)
            seq_results.append(r)
            print(
                f"  [{i+1:2d}/10] wall={r['wall_s']:.2f}s"
                f"  dur={r['audio_duration_s']:.2f}s"
                f"  rtf={r['rtf']:.3f}  bytes={r['audio_bytes']}"
            )
    seq_total = time.perf_counter() - t0
    print(f"  Sequential total: {seq_total:.2f}s for 10 requests")

    # Concurrent batch of 50 (4 at a time)
    print(f"\n--- Concurrent (all {n}, 4 parallel) ---")
    conc_results = []
    batch_size = 4
    async with aiohttp.ClientSession() as session:
        t0 = time.perf_counter()
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch_texts = MIXED_SENTENCES[batch_start:batch_end]
            tasks = [
                generate_single(session, url, text, batch_start + j, OUTPUT_DIR, ref_audio_bytes)
                for j, text in enumerate(batch_texts)
            ]
            results = await asyncio.gather(*tasks)
            conc_results.extend(results)
            done = min(batch_end, n)
            batch_walls = [r["wall_s"] for r in results]
            print(f"  [{done:2d}/{n}] batch max_wall={max(batch_walls):.2f}s")
        conc_total = time.perf_counter() - t0
    print(f"  Concurrent total: {conc_total:.2f}s for {n} requests")

    # Summary
    ok = [r for r in conc_results if r["status"] == 200]
    fail = [r for r in conc_results if r["status"] != 200]

    walls = [r["wall_s"] for r in ok]
    durations = [r["audio_duration_s"] for r in ok]
    rtfs = [r["rtf"] for r in ok]

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {len(ok)}/{n} success, {len(fail)} failed")
    print(f"  Total time: {conc_total:.2f}s")
    print(f"  Throughput: {n / conc_total:.2f} req/s")
    print("")
    print("  Audio duration per sentence:")
    print(f"    Mean:   {sum(durations)/len(durations):.2f}s")
    print(f"    Total:  {sum(durations):.1f}s of audio generated")
    print("")
    print("  RTF (wall_time / audio_duration):")
    print(f"    Mean:   {sum(rtfs)/len(rtfs):.3f}")
    print(f"    Median: {sorted(rtfs)[len(rtfs)//2]:.3f}")
    print(f"    Min:    {min(rtfs):.3f}")
    print(f"    Max:    {max(rtfs):.3f}")
    print("")
    print("  Wall time per request:")
    print(f"    Mean:   {sum(walls)/len(walls):.2f}s")
    print(f"    Median: {sorted(walls)[len(walls)//2]:.2f}s")
    print(f"    Min:    {min(walls):.2f}s")
    print(f"    Max:    {max(walls):.2f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(main())
