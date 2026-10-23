"""Build the files of Collar City Bikes, a made-up bike share in Troy, NY.

They are messy on purpose, the way real company files are:
  - the refund policy exists in two versions, and only the newer one is right
  - the ridership numbers, the station table and the truck route exist only as pictures
  - the winter hours are on a scanned flyer, a page with no text at all
  - a rider's complaint exists only as a voicemail, and the dock reset steps only in a video
  - one answer needs facts from four files: an incident, a memo, the board minutes and the directory

Run: python data/make_documents.py   (the voicemail uses the speech voice built into Windows)
"""

import io
import json
import subprocess
from pathlib import Path

import pymupdf as fitz
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
DOCS = HERE / "docs"
PICTURES = HERE / "pictures"
COMPANY = "Collar City Bikes"

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
RIDES_2025 = [3120, 2890, 4410, 7650, 10980, 12870, 13950, 14280, 11630, 8240, 4970, 3310]
STATIONS = [  # name, docks, rides in 2025
    ("Monument Square", 24, 21400),
    ("RPI Student Union", 30, 19800),
    ("Riverfront Park", 18, 16950),
    ("Troy Atrium", 16, 11200),
    ("Green Island Bridge", 14, 9300),
    ("Lansingburgh", 12, 7450),
]
TRUCK_ROUTE = ["Depot", "Monument Square", "RPI Student Union", "Lansingburgh", "Riverfront Park", "Depot"]

STYLE = """
body { font-family: sans-serif; font-size: 10.5pt; line-height: 1.45; color: #202124; }
h1 { font-size: 17pt; margin: 0 0 4pt 0; }
h2 { font-size: 12.5pt; margin: 12pt 0 4pt 0; }
.meta { color: #5f6368; font-size: 9pt; margin-bottom: 10pt; }
table { border-collapse: collapse; }
td, th { border: 1px solid #9aa0a6; padding: 3pt 6pt; font-size: 9.5pt; }
"""


# ---------------------------------------------------------------- the pictures

def ridership_chart():
    """Monthly rides in 2025. The numbers live only in this chart."""
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=150)
    bars = ax.bar(MONTHS, RIDES_2025, color="#1a73e8")
    bars[RIDES_2025.index(max(RIDES_2025))].set_color("#188038")
    for bar, rides in zip(bars, RIDES_2025):
        ax.text(bar.get_x() + bar.get_width() / 2, rides + 250, f"{rides:,}", ha="center", fontsize=7)
    ax.set_title("Figure 1. Rides per month, 2025")
    ax.set_ylabel("Rides")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(PICTURES / "ridership-2025.png")
    plt.close(fig)


def station_table():
    """The station table is pasted into the report as a screenshot, so it has no text layer."""
    fig, ax = plt.subplots(figsize=(6.5, 2.6), dpi=150)
    ax.axis("off")
    rows = [[name, str(docks), f"{rides:,}"] for name, docks, rides in STATIONS]
    table = ax.table(cellText=rows, colLabels=["Station", "Docks", "Rides in 2025"], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax.set_title("Table 1. Our six busiest stations", fontsize=10)
    fig.tight_layout()
    fig.savefig(PICTURES / "stations-2025.png")
    plt.close(fig)


def truck_route():
    """The rebalancing truck's loop, drawn as a diagram."""
    fig, ax = plt.subplots(figsize=(7, 2.4), dpi=150)
    ax.axis("off")
    xs = np.linspace(0.06, 0.94, len(TRUCK_ROUTE))
    for i, (x, stop) in enumerate(zip(xs, TRUCK_ROUTE)):
        ax.add_patch(plt.Circle((x, 0.5), 0.045, color="#34a853" if stop == "Depot" else "#1a73e8"))
        ax.text(x, 0.5, str(i) if 0 < i < len(TRUCK_ROUTE) - 1 else "D", color="white", ha="center", va="center", fontsize=9, weight="bold")
        ax.text(x, 0.26, stop, ha="center", fontsize=7.5)
        if i:
            ax.annotate("", xy=(x - 0.05, 0.5), xytext=(xs[i - 1] + 0.05, 0.5), arrowprops=dict(arrowstyle="->", color="#5f6368"))
    ax.set_title("Figure 2. Rebalancing truck, morning loop (starts 5:30 am)", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(PICTURES / "truck-route.png", bbox_inches="tight")
    plt.close(fig)


def winter_flyer():
    """A flyer someone scanned crooked. The page is one picture: no text to extract."""
    fig = plt.figure(figsize=(6.2, 8), dpi=110)
    fig.patch.set_facecolor("#f1efe8")
    lines = [
        ("WINTER HOURS", 30, "bold"),
        ("", 10, "normal"),
        ("1 December to 28 February", 17, "bold"),
        ("Bikes can be unlocked", 15, "normal"),
        ("6:00 am to 9:00 pm, every day", 17, "bold"),
        ("", 10, "normal"),
        ("During a city snow emergency", 13, "normal"),
        ("all stations close until", 13, "normal"),
        ("the emergency is lifted.", 13, "normal"),
        ("", 10, "normal"),
        ("E-bikes stay in the depot", 13, "normal"),
        ("when it is below -10 C.", 13, "normal"),
        ("", 10, "normal"),
        (COMPANY, 12, "bold"),
    ]
    y = 0.9
    for text, size, weight in lines:
        fig.text(0.5, y, text, ha="center", fontsize=size, weight=weight, rotation=1.6, color="#2b2b2b", family="serif")
        y -= 0.06
    fig.savefig(PICTURES / "winter-flyer.png", facecolor=fig.get_facecolor())
    plt.close(fig)

    # Scanner noise: speckles and a grey edge, with a fixed seed so every build is the same.
    from matplotlib.image import imread, imsave
    image = imread(PICTURES / "winter-flyer.png")[:, :, :3]
    noise = np.random.default_rng(7).normal(0, 0.04, image.shape[:2])
    image = np.clip(image + noise[:, :, None], 0, 1)
    image[:, :18] *= 0.82
    imsave(PICTURES / "winter-flyer.png", image)


# ---------------------------------------------------------------- the documents

def write_pdf(name, title, date, kind, pages, keywords=""):
    """Each page is a small piece of HTML. A header and a footer go on every page, like real files."""
    buffer = io.BytesIO()
    writer = fitz.DocumentWriter(buffer)
    paper = fitz.paper_rect("a4")
    for html in pages:
        story = fitz.Story(html=f"<style>{STYLE}</style>{html}", archive=fitz.Archive(str(PICTURES)))
        more = True
        while more:
            device = writer.begin_page(paper)
            more, _ = story.place(paper + (54, 64, -54, -60))
            story.draw(device)
            writer.end_page()
    writer.close()

    pdf = fitz.open("pdf", buffer.getvalue())
    for number, page in enumerate(pdf, start=1):
        page.insert_text((54, 38), f"{COMPANY} · {kind} · internal", fontsize=7.5, color=(0.45, 0.45, 0.45))
        page.insert_text((470, 812), f"Page {number} of {pdf.page_count}", fontsize=7.5, color=(0.45, 0.45, 0.45))
    pdf.set_metadata({"title": title, "author": COMPANY, "subject": kind, "keywords": keywords, "creationDate": date})
    pdf.save(DOCS / name, garbage=3, deflate=True)
    print(f"  {name:42} {pdf.page_count} page(s)")
    pdf.close()


def documents():
    write_pdf(
        "2025-annual-ridership-report.pdf", "2025 Annual Ridership Report", "D:20260310", "Report",
        [
            """<h1>2025 Annual Ridership Report</h1>
            <p class="meta">Published 10 March 2026 · Planning team</p>
            <h2>Summary</h2>
            <p>2025 was our busiest year so far. Riders took close to one hundred thousand trips,
            up from 71,000 in 2024. The summer months carried the year: see Figure 1 for the month by month numbers.</p>
            <p>Growth came mostly from students and from commuters crossing the river. The two stations
            near campus and downtown were the busiest, as Table 1 shows. Lansingburgh grew the fastest
            from a small base.</p>
            <p><img src="ridership-2025.png" width="480"/></p>
            <h2>Stations</h2>
            <p>We ran 22 stations at the end of 2025. Table 1 lists the six busiest, with the number of
            docks at each.</p>
            <p><img src="stations-2025.png" width="440"/></p>""",
            """<h2>Operations</h2>
            <p>Every morning a rebalancing truck moves bikes from full stations to empty ones before the
            commute. Figure 2 shows the loop it drives. The route was changed in October 2025 after
            complaints that Lansingburgh was empty by 7 am.</p>
            <p><img src="truck-route.png" width="480"/></p>
            <h2>What we will do in 2026</h2>
            <p>Add e-bikes at the three busiest stations, open four stations in Albany, and keep the
            winter service running. Winter opening hours are on the flyer posted at every station.</p>""",
        ],
        keywords="ridership, stations, rebalancing",
    )

    write_pdf(
        "refund-policy-2024.pdf", "Refund Policy (February 2024)", "D:20240201", "Policy",
        [
            """<h1>Refund Policy</h1>
            <p class="meta">Version 1 · In force from 1 February 2024 · Customer care</p>
            <h2>Faulty bikes</h2>
            <p>If a bike is faulty, for example a flat tyre, broken brakes or a loose chain, report it in the app
            within 30 minutes of unlocking it. We refund the ride as account credit. Refunds are not paid back
            to cards.</p>
            <h2>Memberships</h2>
            <p>Annual memberships cost $90 and can be refunded within 14 days of purchase if no rides were taken.</p>
            <h2>Lost or stolen bikes</h2>
            <p>A bike that is not returned within 24 hours is treated as lost. The member is charged $1,200.</p>""",
        ],
        keywords="refund, faulty bike, membership",
    )

    write_pdf(
        "refund-policy-2026.pdf", "Refund Policy (January 2026)", "D:20260115", "Policy",
        [
            """<h1>Refund Policy</h1>
            <p class="meta">Version 2 · In force from 15 January 2026 · Customer care ·
            This version replaces the policy of 1 February 2024.</p>
            <h2>Faulty bikes</h2>
            <p>Report a faulty bike in the app within 60 minutes of unlocking it. We refund the full ride to the
            original payment card within 5 business days, and add a $5 ride credit to say sorry.</p>
            <h2>Memberships</h2>
            <p>Annual memberships cost $95. They can be refunded within 14 days of purchase if no rides were taken.</p>
            <h2>Lost or stolen bikes</h2>
            <p>A bike that is not returned within 24 hours is treated as lost. The member is charged $950, down
            from $1,200, because most lost bikes are now found by their GPS tracker.</p>""",
        ],
        keywords="refund, faulty bike, membership",
    )

    write_pdf(
        "incident-report-IR-2026-031.pdf", "Incident report IR-2026-031", "D:20260312", "Incident",
        [
            """<h1>Incident report IR-2026-031</h1>
            <p class="meta">Filed 12 March 2026 · Field operations</p>
            <table>
            <tr><th>Where</th><th>When</th><th>Reported by</th><th>Technician</th></tr>
            <tr><td>Monument Square station, dock 7</td><td>12 March 2026, 08:40</td><td>A rider, through the app</td><td>Jamal Ortiz</td></tr>
            </table>
            <h2>What happened</h2>
            <p>The dock would not release bike 4471. The rider was charged for a ride they could not take.
            Jamal Ortiz found a worn locking pin. A replacement pin is on order, and dock 7 stays out of
            service until it is fitted.</p>
            <h2>Status</h2>
            <p>Open. A repaired dock goes back into service only after the station manager signs off the repair.
            The rider's refund follows the refund policy in force on the day of the incident.</p>""",
        ],
        keywords="incident, dock, Monument Square",
    )

    write_pdf(
        "station-operations-memo-2026.pdf", "Station operations memo", "D:20260105", "Memo",
        [
            """<h1>Station operations for 2026</h1>
            <p class="meta">5 January 2026 · From the operations director to all field staff</p>
            <p>Every station has one station manager. The station manager is the person who signs off any
            repaired dock before it goes back into service. Technicians report to the station manager of the
            station they are working at.</p>
            <h2>Who manages what</h2>
            <table>
            <tr><th>Station</th><th>Station manager</th></tr>
            <tr><td>Monument Square</td><td>Priya Raman</td></tr>
            <tr><td>Riverfront Park</td><td>Marcus Lee</td></tr>
            <tr><td>Lansingburgh</td><td>Marcus Lee</td></tr>
            <tr><td>RPI Student Union</td><td>Elena Petrova</td></tr>
            <tr><td>Green Island Bridge</td><td>Elena Petrova</td></tr>
            <tr><td>Troy Atrium</td><td>Daniel Kim</td></tr>
            </table>
            <p>Changes to this list are decided by the board and recorded in its minutes.</p>""",
        ],
        keywords="station manager, sign-off, operations",
    )

    write_pdf(
        "board-minutes-2026-04-22.pdf", "Board meeting minutes, 22 April 2026", "D:20260422", "Minutes",
        [
            """<h1>Board meeting minutes</h1>
            <p class="meta">22 April 2026 · Troy · Seven of seven board members present</p>
            <h2>1. Albany expansion</h2>
            <p>The board approved four stations in Albany, opening in September 2026, with a budget of $310,000.</p>
            <h2>2. Appointments</h2>
            <p>Priya Raman is appointed Head of Operations from 1 May 2026. From the same date Daniel Kim
            takes over as station manager of Monument Square, in addition to Troy Atrium.</p>
            <h2>3. E-bikes</h2>
            <p>Sixty e-bikes are ordered for delivery in June. They will be placed at Monument Square,
            RPI Student Union and Riverfront Park first.</p>""",
        ],
        keywords="board, appointments, Albany",
    )

    write_pdf(
        "staff-directory-2026.pdf", "Staff directory", "D:20260201", "Directory",
        [
            """<h1>Staff directory</h1>
            <p class="meta">February 2026 · People team · Phone numbers are direct lines</p>
            <table>
            <tr><th>Name</th><th>Role</th><th>Phone</th><th>Email</th></tr>
            <tr><td>Priya Raman</td><td>Station manager</td><td>518-555-0142</td><td>priya.raman@collarcitybikes.example</td></tr>
            <tr><td>Marcus Lee</td><td>Station manager</td><td>518-555-0177</td><td>marcus.lee@collarcitybikes.example</td></tr>
            <tr><td>Elena Petrova</td><td>Station manager</td><td>518-555-0123</td><td>elena.petrova@collarcitybikes.example</td></tr>
            <tr><td>Daniel Kim</td><td>Station manager</td><td>518-555-0158</td><td>daniel.kim@collarcitybikes.example</td></tr>
            <tr><td>Jamal Ortiz</td><td>Technician</td><td>518-555-0190</td><td>jamal.ortiz@collarcitybikes.example</td></tr>
            <tr><td>Grace Okafor</td><td>Customer care lead</td><td>518-555-0101</td><td>grace.okafor@collarcitybikes.example</td></tr>
            </table>""",
        ],
        keywords="directory, phone, email",
    )

    write_pdf(
        "winter-hours-flyer.pdf", "Winter hours flyer (scanned)", "D:20251120", "Flyer",
        ["""<p><img src="winter-flyer.png" width="480"/></p>"""],
        keywords="winter, opening hours",
    )


VOICEMAIL = (
    "Hi, this is Sam Carter, member number 2 0 4 1 7, calling on Monday the sixth of April. "
    "On Saturday the e-bike I took from Riverfront Park lost all power halfway across the Green Island Bridge, "
    "and I had to push it the rest of the way. The bike number was 9 0 1 2. "
    "I would like a refund for that ride, please. You can call me back on 5 1 8, 5 5 5, 0 1 9 9. Thanks, bye."
)

TRAINING_CARDS = [  # each card stays on screen for six seconds
    ("Dock training", "Freeing a bike that is stuck in a dock"),
    ("Step 1", "Hold the red reset button on the dock for 5 seconds"),
    ("Step 2", "Wait up to 30 seconds for the dock light to turn green"),
    ("Step 3", "Still red? Tag the dock OUT OF SERVICE and call the technician on duty"),
]


def sidecar(path: Path, title: str, kind: str, date: str):
    """Audio and video files carry no title or date of their own, so they get a small JSON file beside them."""
    path.with_suffix(".json").write_text(json.dumps({"title": title, "kind": kind, "date": date}, indent=1))


def voicemail():
    path = DOCS / "rider-voicemail-2026-04-06.wav"
    speak = (
        "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SelectVoice('Microsoft Zira Desktop'); $s.Rate = -1; $s.SetOutputToWaveFile('{path}'); "
        f"$s.Speak('{VOICEMAIL}'); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", speak], check=True)
    sidecar(path, "Voicemail from a rider, 6 April 2026", "Voicemail", "2026-04-06")
    print(f"  {path.name:42} {path.stat().st_size // 1024} KB")


def training_video():
    frames = []
    for heading, line in TRAINING_CARDS:
        fig = plt.figure(figsize=(6.4, 3.6), dpi=100)
        fig.patch.set_facecolor("#202124")
        fig.text(0.5, 0.62, heading, ha="center", fontsize=26, weight="bold", color="#fbbc04")
        fig.text(0.5, 0.38, line, ha="center", fontsize=13, color="white", wrap=True)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        frames += [frame] * 12  # 6 seconds at 2 frames a second
    path = DOCS / "dock-training-video.mp4"
    imageio.mimsave(path, frames, fps=2, codec="libx264", macro_block_size=1)
    sidecar(path, "Dock training video", "Training", "2026-02-10")
    print(f"  {path.name:42} {len(TRAINING_CARDS) * 6} seconds")


if __name__ == "__main__":
    DOCS.mkdir(parents=True, exist_ok=True)
    PICTURES.mkdir(parents=True, exist_ok=True)
    ridership_chart()
    station_table()
    truck_route()
    winter_flyer()
    print("Writing the files:")
    documents()
    voicemail()
    training_video()
