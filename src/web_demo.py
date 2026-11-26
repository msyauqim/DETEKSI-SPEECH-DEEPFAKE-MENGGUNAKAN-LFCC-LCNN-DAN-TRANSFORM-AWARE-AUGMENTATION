# src/web_demo.py
# Demo web lokal: rekam suara lewat mic ATAU unggah berkas audio, lalu lihat
# bagaimana dua detektor dari skripsi ini menilainya -- baseline (clean-only)
# dan augmented (transform-aware) -- pada SATU sumbu bersama asli -> deepfake.
#
# Tiap skema dinilai oleh ketiga seed-nya (2026, 2027, 2028):
#   - titik  = fake_score satu seed; penuh bila seed itu menilai deepfake
#              (melewati val_threshold-nya sendiri), kosong bila menilai asli
#   - garis  = rerata fake_score ketiga seed
#   - vonis  = suara mayoritas ketiga seed
#
# Mesin penilai memakai fungsi dari nilai_rekaman.py, jadi jalur fiturnya
# identik dengan evaluasi skripsi.
#
# Warna mengikuti gambar skripsi (Okabe-Ito biru/jingga), divalidasi untuk
# buta warna: terang #0072B2/#D55E00, gelap #2F8FD0/#DD6E26 -- seluruh cek lolos.
#
#   .venv/bin/python src/web_demo.py
#   lalu buka http://127.0.0.1:7860
import html as H
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# checkpoints/ dan statistik CMVN dirujuk relatif ke akar proyek, jadi demo
# tetap jalan walau dipanggil dari folder lain (mis. ~/Skripsi).
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

from nilai_rekaman import jendela, ke_wav, muat, nilai

SKEMA = {
    "baseline":  ("Baseline", "dilatih pada rekaman bersih"),
    "augmented": ("Augmented", "dilatih dengan derau, gema, dan gangguan kanal"),
}
SEEDS = (2026, 2027, 2028)

print("memuat 6 model ...", flush=True)
# model dari eksperimen v2 (validasi terpisah)
VERSI = os.environ.get("VERSI_MODEL", "v2_validasi_terpisah")
MODEL = {sk: [muat(f"checkpoints/{VERSI}/{sk}_seed{s}.pt") for s in SEEDS] for sk in SKEMA}
print("siap. buka http://127.0.0.1:7860", flush=True)


def koma(x, n=3):
    return f"{x:.{n}f}".replace(".", ",")


def skor_teks(v):
    """Skor 0..1. Model baseline sering jenuh (mis. 3e-13), yang bila dibulatkan
    tampil 0,000 seolah tidak ada nilai. Tulis sebagai batas agar terbaca
    'sangat yakin'."""
    if v < 0.0005:
        return "< 0,001"
    if v > 0.9995:
        return "> 0,999"
    return koma(v)


def vonis_kata(palsu):
    return "deepfake" if palsu else "asli"


# --------------------------------------------------------------- tampilan hasil
def kosong(pesan):
    return f"<div class='dd-kosong'>{pesan}</div>"


def baris_skema(sk, skor, putusan):
    nama, ket = SKEMA[sk]
    rerata = sum(skor) / len(skor)
    n_palsu = sum(putusan)
    # Titik yang nilainya berdekatan (< 2,5% sumbu) akan menumpuk dan
    # menyembunyikan seed. Tiap gugus disebar menyamping; di dekat tepi sumbu
    # disebar ke arah dalam agar tidak terpotong.
    geser = {}
    urut = sorted(range(len(skor)), key=lambda i: skor[i])
    gugus = []
    for i in urut:
        if gugus and skor[i] - skor[gugus[-1][-1]] < 0.025:
            gugus[-1].append(i)
        else:
            gugus.append([i])
    for g in gugus:
        v = sum(skor[i] for i in g) / len(g)
        n = len(g)
        if v < 0.05:
            pola = [7 + 15 * k for k in range(n)]
        elif v > 0.95:
            pola = [-7 - 15 * k for k in range(n)][::-1]
        else:
            pola = [15 * (k - (n - 1) / 2) for k in range(n)]
        for i, dx in zip(g, pola):
            geser[i] = dx
    titik = []
    for i, (s, v, p) in enumerate(zip(SEEDS, skor, putusan)):
        lbl = f"Seed {s}: {skor_teks(v)}, menilai {vonis_kata(p)}"
        titik.append(f"<span class='dd-titik {'penuh' if p else 'kosong'}' "
                     f"style='left:calc({v*100:.2f}% + {geser[i]:.0f}px)' title='{lbl}' "
                     f"aria-label='{lbl}'></span>")
    return f"""
    <div class="dd-baris dd-{sk}">
      <div class="dd-atas">
        <div class="dd-nama"><span class="dd-tanda"></span>
          <span><b>{nama}</b><small>{ket}</small></span></div>
        <div class="dd-vonis"><b>{vonis_kata(n_palsu >= 2).capitalize()}</b>
          <small>{n_palsu} dari 3 seed, rerata {skor_teks(rerata)}</small></div>
      </div>
      <div class="dd-jalur" role="img"
           aria-label="{nama}: rerata {skor_teks(rerata)}, {n_palsu} dari 3 seed menilai deepfake">
        <span class="dd-rerata" style="left:clamp(4px, {rerata*100:.2f}%, calc(100% - 4px))"></span>
        {''.join(titik)}
      </div>
    </div>"""


def kalimat(hasil):
    b, a = (hasil[k]["mayoritas"] for k in ("baseline", "augmented"))
    if b == a:
        return f"Kedua detektor menilai rekaman ini {vonis_kata(b)}."
    return (f"Kedua detektor berbeda pendapat: baseline menilainya {vonis_kata(b)}, "
            f"augmented menilainya {vonis_kata(a)}.")


def tabel(hasil):
    rows = []
    for sk in SKEMA:
        for s, v, t, p in zip(SEEDS, hasil[sk]["skor"], hasil[sk]["ambang"], hasil[sk]["putusan"]):
            rows.append(f"<tr><td>{SKEMA[sk][0]}</td><td>{s}</td><td>{skor_teks(v)}</td>"
                        f"<td>{koma(t)}</td><td>{vonis_kata(p)}</td></tr>")
    return ("<details class='dd-angka'><summary>Lihat angka per seed</summary>"
            "<table><thead><tr><th>Detektor</th><th>Seed</th><th>Skor</th>"
            "<th>Ambang</th><th>Penilaian</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>"
            "<p>Skor melewati ambang berarti seed itu menilai deepfake. Ambang tiap "
            "seed diambil dari data validasi saat pelatihan.</p></details>")


def analisis(path):
    if not path:
        return kosong("Belum ada rekaman. Rekam suaramu atau unggah berkas audio, "
                      "lalu tekan <b>Analisis rekaman</b>.")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            potong, durasi, _ = jendela(ke_wav(path, tmp))
        except Exception:
            return kosong("Berkas ini tidak bisa dibaca. Gunakan format wav, m4a, mp3, "
                          "atau aiff.")
    if durasi < 1.0:
        return kosong(f"Rekaman hanya {koma(durasi, 1)} detik. Rekam minimal beberapa "
                      "detik, idealnya 8 sampai 15 detik.")

    hasil = {}
    for sk, daftar in MODEL.items():
        skor, ambang, putusan = [], [], []
        for model, lfcc, thr in daftar:
            f = float(nilai(model, lfcc, potong).mean())
            skor.append(f); ambang.append(thr); putusan.append(f >= thr)
        hasil[sk] = {"skor": skor, "ambang": ambang, "putusan": putusan,
                     "mayoritas": sum(putusan) >= 2}

    catatan = (f"Rekaman {koma(durasi, 1)} detik, dinilai dalam {len(potong)} potongan "
               f"4 detik.")
    if durasi < 4.0:
        catatan += " Rekaman di bawah 4 detik ditambal hening; 8 sampai 15 detik lebih stabil."

    return f"""
    <div class="dd-hasil">
      <p class="dd-kalimat">{H.escape(kalimat(hasil))}</p>
      <p class="dd-meta">{catatan}</p>
      <div class="dd-skala" aria-hidden="true">
        <i style="left:0">0</i><i style="left:25%">0,25</i><i style="left:50%">0,5</i>
        <i style="left:75%">0,75</i><i style="left:100%">1</i></div>
      {baris_skema("baseline", **{k: hasil["baseline"][k] for k in ("skor", "putusan")})}
      {baris_skema("augmented", **{k: hasil["augmented"][k] for k in ("skor", "putusan")})}
      <div class="dd-ujung" aria-hidden="true">
        <em>&larr; lebih yakin asli</em><em>lebih yakin deepfake &rarr;</em></div>
      <p class="dd-legenda">
        <span class="dd-item"><span class="dd-baseline dd-titik penuh contoh"></span><span class="dd-augmented dd-titik penuh contoh"></span>seed menilai deepfake</span>
        <span class="dd-item"><span class="dd-baseline dd-titik kosong contoh"></span><span class="dd-augmented dd-titik kosong contoh"></span>seed menilai asli</span>
        <span class="dd-item"><span class="dd-rerata contoh"></span>rerata ketiga seed</span></p>
      {tabel(hasil)}
    </div>"""


# --------------------------------------------------------------- gaya
HEAD = ("<link rel='preconnect' href='https://fonts.googleapis.com'>"
        "<link rel='stylesheet' href='https://fonts.googleapis.com/css2?"
        "family=Bricolage+Grotesque:opsz,wght@12..96,500..800&"
        "family=Instrument+Sans:ital,wght@0,400..700;1,400&display=swap'>")

CSS = """
:root{--dd-kertas:#EDF0EC;--dd-muka:#F7F9F6;--dd-tinta:#17201B;--dd-redup:#56625B;
  --dd-garis:#CBD3CD;--dd-jalur:#E2E7E3;--dd-base:#0072B2;--dd-aug:#D55E00;
  --dd-judul:'Bricolage Grotesque',ui-sans-serif,system-ui,sans-serif;
  --dd-isi:'Instrument Sans',ui-sans-serif,system-ui,sans-serif}
.dark{--dd-kertas:#161C19;--dd-muka:#1C2320;--dd-tinta:#E4EAE6;--dd-redup:#98A39D;
  --dd-garis:#33403A;--dd-jalur:#26302B;--dd-base:#2F8FD0;--dd-aug:#DD6E26}

footer{display:none!important}
.gradio-container{max-width:1080px!important;margin:0 auto!important;
  font-family:var(--dd-isi)!important;background:var(--dd-kertas)!important}
.gradio-container *{font-family:inherit}

/* kepala halaman */
#dd-kepala{padding:28px 0 8px}
#dd-kepala h1{font-family:var(--dd-judul);font-weight:800;font-size:clamp(2.2rem,5vw,3.4rem);
  line-height:1.02;letter-spacing:-.025em;color:var(--dd-tinta);margin:0 0 14px;
  font-variation-settings:'opsz' 96}
#dd-kepala p{font-size:1.06rem;line-height:1.55;color:var(--dd-redup);max-width:62ch;margin:0}

/* panel hasil */
.dd-hasil,.dd-kosong{background:var(--dd-muka);border:1px solid var(--dd-garis);
  border-radius:14px;padding:24px 26px;color:var(--dd-tinta)}
.dd-kosong{color:var(--dd-redup);line-height:1.55;min-height:180px;display:flex;align-items:center}
.dd-kalimat{font-family:var(--dd-judul);font-weight:650;font-size:1.45rem;line-height:1.25;
  letter-spacing:-.01em;margin:0 0 6px;max-width:32ch}
.dd-meta{color:var(--dd-redup);font-size:.92rem;margin:0 0 26px}

/* sumbu bersama: tiap detektor = baris judul + jalur selebar penuh */
.dd-baris{padding:16px 0 18px;border-top:1px solid var(--dd-garis)}
.dd-baris+.dd-baris{border-bottom:1px solid var(--dd-garis)}
.dd-atas{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin-bottom:12px}
.dd-vonis{text-align:right}
.dd-skala{position:relative;height:20px;margin:0 7px 6px}
.dd-skala i{position:absolute;transform:translateX(-50%);font-style:normal;font-size:.78rem;
  color:var(--dd-redup);font-variant-numeric:tabular-nums}
.dd-ujung{display:flex;justify-content:space-between;margin-top:10px}
.dd-ujung em{font-style:normal;font-size:.82rem;color:var(--dd-redup)}

.dd-nama{display:flex;gap:10px;align-items:flex-start}
.dd-nama b{display:block;font-size:1.02rem}
.dd-nama small,.dd-vonis small{display:block;color:var(--dd-redup);font-size:.8rem;line-height:1.35;margin-top:2px}
.dd-tanda{width:10px;height:10px;border-radius:50%;background:var(--dd-c);margin-top:6px;flex:none}
.dd-vonis b{font-family:var(--dd-judul);font-weight:700;font-size:1.15rem}
.dd-baseline{--dd-c:var(--dd-base)}
.dd-augmented{--dd-c:var(--dd-aug)}

.dd-jalur{position:relative;height:30px;margin:0 7px;border-radius:15px;background:
  linear-gradient(90deg,transparent calc(25% - .5px),var(--dd-garis) calc(25% - .5px),var(--dd-garis) calc(25% + .5px),transparent calc(25% + .5px),
  transparent calc(50% - .5px),var(--dd-garis) calc(50% - .5px),var(--dd-garis) calc(50% + .5px),transparent calc(50% + .5px),
  transparent calc(75% - .5px),var(--dd-garis) calc(75% - .5px),var(--dd-garis) calc(75% + .5px),transparent calc(75% + .5px)),
  var(--dd-jalur)}
.dd-rerata{position:absolute;top:3px;bottom:3px;width:4px;margin-left:-2px;border-radius:2px;
  background:var(--dd-tinta);animation:dd-geser .7s cubic-bezier(.2,.8,.2,1)}
.dd-titik{position:absolute;top:50%;width:14px;height:14px;margin:-7px 0 0 -7px;border-radius:50%;
  box-shadow:0 0 0 2px var(--dd-muka);animation:dd-geser .7s cubic-bezier(.2,.8,.2,1)}
.dd-titik.penuh{background:var(--dd-c)}
.dd-titik.kosong{background:color-mix(in srgb,var(--dd-c) 30%,var(--dd-muka));
  border:2.5px solid var(--dd-c);box-sizing:border-box}
@keyframes dd-geser{from{left:0}}
@media (prefers-reduced-motion:reduce){.dd-rerata,.dd-titik{animation:none}}

.dd-legenda{display:flex;flex-wrap:wrap;align-items:center;gap:8px 20px;margin:20px 0 0;
  font-size:.84rem;color:var(--dd-redup)}
.dd-item{display:inline-flex;align-items:center;gap:8px;white-space:nowrap}
.dd-legenda .contoh{position:relative;top:auto;left:auto;margin:0;animation:none;display:inline-block}
.dd-legenda .dd-rerata.contoh{--dd-c:var(--dd-tinta)}
.dd-item .contoh+.contoh{margin-left:-4px}
.dd-legenda .dd-rerata.contoh{width:4px;height:16px}

.dd-angka{margin-top:18px;font-size:.88rem}
.dd-angka summary{cursor:pointer;color:var(--dd-tinta);font-weight:700}
.dd-angka table{border-collapse:collapse;margin:12px 0 8px;font-variant-numeric:tabular-nums;width:100%}
.dd-angka table,.dd-angka thead,.dd-angka tbody,.dd-angka tr,.dd-angka th,.dd-angka td{
  border:0!important;background:transparent!important;outline:0!important}
.dd-angka th,.dd-angka td{text-align:left;padding:7px 14px 7px 0;
  border-bottom:1px solid var(--dd-garis)!important;color:var(--dd-tinta)}
.dd-angka th{color:var(--dd-redup);font-weight:400}
.dd-angka p{color:var(--dd-redup);margin:0;line-height:1.5}

/* kolom masukan */
#dd-tombol{font-family:var(--dd-judul)!important;font-weight:700;font-size:1.05rem;
  border-radius:12px!important;min-height:52px}
#dd-tombol:focus-visible{outline:3px solid var(--dd-base);outline-offset:3px}
#dd-petunjuk{color:var(--dd-redup);font-size:.88rem;line-height:1.5}

/* catatan bawah */
#dd-catatan{margin-top:28px;padding-top:22px;border-top:1px solid var(--dd-garis);
  flex-wrap:wrap!important;gap:24px 40px!important}
#dd-catatan>*{flex:1 1 300px!important;min-width:0!important}
#dd-catatan h3{font-family:var(--dd-judul);font-weight:700;font-size:1.05rem;margin:0 0 6px;color:var(--dd-tinta)}
#dd-catatan p{color:var(--dd-redup);font-size:.92rem;line-height:1.6;max-width:60ch;margin:0}

@media (max-width:720px){
  .dd-atas{flex-direction:column;gap:8px}.dd-vonis{text-align:left}
  .dd-hasil{padding:20px 18px}
}
"""

TEMA = gr.themes.Base(
    font=[gr.themes.GoogleFont("Instrument Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#EDF0EC", body_background_fill_dark="#161C19",
    body_text_color="#17201B", body_text_color_dark="#E4EAE6",
    body_text_color_subdued="#56625B", body_text_color_subdued_dark="#98A39D",
    block_background_fill="#F7F9F6", block_background_fill_dark="#1C2320",
    block_border_color="#CBD3CD", block_border_color_dark="#33403A",
    block_radius="14px",
    block_label_background_fill="#E2E7E3", block_label_background_fill_dark="#26302B",
    block_label_text_color="#17201B", block_label_text_color_dark="#E4EAE6",
    button_primary_background_fill="#17201B", button_primary_background_fill_dark="#E4EAE6",
    button_primary_background_fill_hover="#2A352F", button_primary_background_fill_hover_dark="#FFFFFF",
    button_primary_text_color="#F7F9F6", button_primary_text_color_dark="#161C19",
)

# --------------------------------------------------------------- tata letak
with gr.Blocks(title="Asli atau deepfake? Demo detektor LFCC-LCNN") as demo:
    gr.HTML("""<div id="dd-kepala"><h1>Asli atau deepfake?</h1>
      <p>Rekam suaramu atau unggah berkas audio. Dua detektor LFCC-LCNN dari skripsi ini
      akan menilainya: baseline yang dilatih pada rekaman bersih, dan augmented yang
      dilatih dengan derau, gema, serta gangguan kanal.</p></div>""")
    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=300):
            audio = gr.Audio(sources=["microphone", "upload"], type="filepath",
                             label="Rekaman")
            tombol = gr.Button("Analisis rekaman", variant="primary", elem_id="dd-tombol")
            gr.HTML("<p id='dd-petunjuk'>Bicara 8 sampai 15 detik, sebaiknya dalam bahasa "
                    "Inggris. Berkas wav, m4a, mp3, dan aiff bisa langsung diunggah.</p>")
        with gr.Column(scale=8, min_width=320):
            hasil = gr.HTML(kosong("Belum ada rekaman. Rekam suaramu atau unggah berkas "
                                   "audio, lalu tekan <b>Analisis rekaman</b>."))
    tombol.click(analisis, inputs=audio, outputs=hasil, api_name="analisis")
    with gr.Row(elem_id="dd-catatan"):
        gr.HTML("""<h3>Cara membaca grafik</h3><p>Kedua detektor diletakkan pada sumbu
          yang sama, dari 0 (yakin asli) sampai 1 (yakin deepfake). Tiap titik adalah satu
          model dari tiga kali pelatihan; titik penuh berarti model itu menilai deepfake.
          Vonis ditentukan suara mayoritas ketiga model.</p>""")
        gr.HTML("""<h3>Batas kemampuan model</h3><p>Model dilatih pada ucapan berbahasa
          Inggris yang direkam di studio. Rekaman mic di ruangan biasa lebih mirip kondisi
          dunia nyata, tempat model ini terukur jauh lebih lemah. Perhatikan letak titiknya,
          bukan vonisnya saja; bahasa selain Inggris belum andal.</p>""")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, css=CSS, theme=TEMA,
                head=HEAD, show_error=True)
