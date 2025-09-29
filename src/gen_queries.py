#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sinh tập truy vấn tiếng Anh cho Day 1 (2k–3k dòng), bám 18 thực thể (Table A1).
Lưu dạng JSONL: {"id": "q_000001", "text": "..."}

Cách chạy:
  python gen_queries.py --n 2500 --out data/synthetic_queries.jsonl --persona 1
"""

import argparse, json, os, random, re
from collections import Counter

# ----------------------------
# 0) Seed để tái lập kết quả
# ----------------------------
random.seed(42)

# ----------------------------
# 1) 18 thực thể theo Appendix A – Table A1
# ----------------------------
ENTITIES = [
    "IntentMovie","IntentTvSeries","Theme","Genre","CastAndCrew","TVSeriesName","MovieName",
    "StreamingService","Recency","Popularity","ReleaseYear","Decade","FreeContent","AudioLanguage",
    "Franchise","Holiday","Sport","Character"
]
# (Nguồn: Appendix A – Table A1: Query Intent Understanding Entities trong bài báo)  # :contentReference[oaicite:2]{index=2}

# ----------------------------
# 2) Từ điển (vocab) tối thiểu
#    Bạn có thể mở rộng tuỳ ý để tăng đa dạng.
# ----------------------------
GENRES = ["horror","romantic comedy","sci-fi","action","drama","thriller","mystery","fantasy",
          "animation","documentary","family","adventure","crime","biopic","war"]
ACTORS = ["Tom Hanks","Leonardo DiCaprio","Scarlett Johansson","Keanu Reeves","Natalie Portman",
          "Denzel Washington","Emma Stone","Ryan Gosling","Jennifer Lawrence","Brad Pitt",
          "Robert Downey Jr.","Meryl Streep"]
DIRECTORS = ["Christopher Nolan","Quentin Tarantino","Greta Gerwig","Steven Spielberg",
             "Martin Scorsese","Denis Villeneuve","James Cameron","Patty Jenkins"]
LANGS = ["Arabic","Bangla","Chinese","English","French","German","Hindi","Italian","Japanese",
         "Korean","Portuguese","Spanish","Turkish","Vietnamese"]  # :contentReference[oaicite:3]{index=3}
YEARS = list(range(1980, 2025))
DECADES = ["80s","90s","2000s","2010s","2020s"]
SERVICES = ["Netflix","HBO Max","Amazon Prime","Apple TV","Hulu","Disney+","Peacock","Paramount+"]  # :contentReference[oaicite:4]{index=4}
SERIES = ["The Marvelous Mrs. Maisel","Stranger Things","Breaking Bad","The Crown","The Mandalorian",
          "The Witcher","Game of Thrones","The Boys"]  # :contentReference[oaicite:5]{index=5}
MOVIES = ["Interstellar","Oppenheimer","Inception","La La Land","Titanic","The Godfather",
          "The Dark Knight","Parasite","Dune","Toy Story"]
FRANCHISES = ["Avengers","Batman","Spider-Man","Star Wars","Harry Potter","Mission Impossible",
              "Fast and Furious","James Bond"]  # :contentReference[oaicite:6]{index=6}
HOLIDAYS = ["Christmas","Thanksgiving","Easter","Halloween","Valentine's Day","New Year"]  # :contentReference[oaicite:7]{index=7}
SPORTS = ["football","basketball","tennis","cricket","baseball","soccer","Formula 1","rugby","golf",
          "hockey","Manchester United","Serena Williams","Rafael Nadal","LeBron James"]  # :contentReference[oaicite:8]{index=8}
CHARACTERS = ["Batman","Asterix","Blippi","Charlie Brown","Sherlock Holmes","James Bond",
              "Spider-Man","Wonder Woman","Superman"]  # :contentReference[oaicite:9]{index=9}

def pick(lst): return random.choice(lst)

# ----------------------------
# 3) Persona flavor (tuỳ chọn)
# ----------------------------
PERSONA_PREFIX = {
    "Merchandiser": ["curate a list of","recommend","suggest","highlight"],
    "Movie Critic": ["critically acclaimed","award-winning","festival favorite","arthouse"],
    "Horror Aficionado": ["genuinely scary","psychological horror","found footage","slasher classics"],
}
def persona_wrap(text: str, enable: bool) -> str:
    if not enable: return text
    # 30% câu có thêm flavor, nhẹ nhàng để không làm hỏng form câu
    if random.random() < 0.30:
        persona = pick(list(PERSONA_PREFIX.keys()))
        token = pick(PERSONA_PREFIX[persona])
        if random.random() < 0.5:
            return f"{token} {text}"
        else:
            return f"{text} ({persona.lower()} pick)"
    return text

# ----------------------------
# 4) Template theo từng thực thể (5+ mẫu mỗi nhóm)
# ----------------------------
def q_genre():
    return pick([
        f"{pick(GENRES)} movies",
        f"best {pick(GENRES)} films",
        f"top {pick(GENRES)} movies in {pick(DECADES)}",
        f"{pick(GENRES)} movies like {pick(MOVIES)}",
        f"{pick(GENRES)} {pick(['movies','films'])} for {pick(['families','date night','weekend binge'])}",
    ])

def q_cast_and_crew():
    return pick([
        f"{pick(ACTORS)} movies",
        f"films directed by {pick(DIRECTORS)}",
        f"best performances by {pick(ACTORS)}",
        f"{pick(ACTORS)} {pick(['filmography','movie list'])}",
        f"top {pick(DIRECTORS)} movies",
    ])

def q_audio_language():
    return pick([
        f"{pick(LANGS)} movies",
        f"{pick(LANGS)} {pick(['films','TV shows'])}",
        f"{pick(['dubbed','subtitled'])} {pick(LANGS)} movies",
        f"{pick(LANGS)} {pick(['cinema classics','new releases'])}",
        f"{pick(['learn','practice'])} {pick(LANGS)} with {pick(['movies','TV series'])}",
    ])

def q_release_year_decade():
    y = pick(YEARS); decade = pick(DECADES)
    t = pick(["movies","TV show","TV series"])
    return pick([
        f"{y} {t}",
        f"{decade} {pick(['movies','shows'])}",
        f"best {t} of {y}",
        f"top {t} from the {decade}",
        f"{t} released in {y}",
    ])

def q_intent_tvseries():
    return pick([
        f"TV series like {pick(SERIES)}",
        f"best {pick(GENRES)} shows",
        f"critically acclaimed TV series",
        f"{pick(['new','recent'])} {pick(GENRES)} shows",
        f"{pick(SERIES)} similar shows",
    ])

def q_intent_movie():
    return pick([
        f"movies like {pick(MOVIES)}",
        f"{pick(['new','recent'])} {pick(GENRES)} movies",
        f"award-winning {pick(GENRES)} films",
        f"{pick(['family-friendly','R-rated','classic'])} {pick(GENRES)} movies",
        f"{pick(MOVIES)} similar movies",
    ])

def q_streaming_service():
    return pick([
        f"{pick(SERVICES)} originals",
        f"best movies on {pick(SERVICES)}",
        f"{pick(SERVICES)} {pick(['new releases','top picks'])}",
        f"{pick(GENRES)} shows on {pick(SERVICES)}",
        f"{pick(SERVICES)} {pick(['documentaries','comedies','dramas'])}",
    ])

def q_recency_popularity_free():
    return pick([
        "newly-released movies",
        "recent TV shows",
        "popular movies",
        "highly-streamed movies",
        "trending shows this week",
        "free movies",
        "free shows",
        "where to watch free movies",
        "best free TV series",
        "free documentaries online",
    ])

def q_franchise():
    return pick([
        f"{pick(FRANCHISES)} movies",
        f"all {pick(FRANCHISES)} films in order",
        f"best {pick(FRANCHISES)} titles",
        f"{pick(FRANCHISES)} watch order",
        f"{pick(FRANCHISES)} spin-offs",
    ])

def q_holiday():
    return pick([
        f"{pick(HOLIDAYS)} movies",
        f"family {pick(HOLIDAYS)} films",
        f"romantic {pick(HOLIDAYS)} movies",
        f"best {pick(HOLIDAYS)} specials",
        f"{pick(HOLIDAYS)} TV episodes",
    ])

def q_sport():
    return pick([
        f"{pick(SPORTS)} movies",
        f"best sports documentaries",
        f"{pick(SPORTS)} highlights and shows",
        f"biopics about {pick(['athletes','coaches'])}",
        f"{pick(['classic','modern'])} {pick(SPORTS)} films",
    ])

def q_character():
    return pick([
        f"{pick(CHARACTERS)} movies",
        f"{pick(CHARACTERS)} animated series",
        f"shows featuring {pick(CHARACTERS)}",
        f"{pick(CHARACTERS)} for kids",
        f"{pick(CHARACTERS)} comics adapted to screen",
    ])

def q_movie_name():
    return pick([
        pick(MOVIES),
        f"movies like {pick(MOVIES)}",
        f"watch {pick(MOVIES)} online",
        f"soundtrack of {pick(MOVIES)}",
        f"{pick(MOVIES)} cast",
    ])

def q_tvseries_name():
    return pick([
        pick(SERIES),
        f"TV series like {pick(SERIES)}",
        f"{pick(SERIES)} season 1",
        f"{pick(SERIES)} cast",
        f"where to watch {pick(SERIES)}",
    ])

# Map entity -> generator
GEN_MAP = {
    "IntentMovie": q_intent_movie,
    "IntentTvSeries": q_intent_tvseries,
    "Theme": lambda: pick([
        "based on true story movies",
        "movies featuring talking animals",
        "post-apocalyptic films",
        "coming-of-age movies",
        "time travel movies"
    ]),
    "Genre": q_genre,
    "CastAndCrew": q_cast_and_crew,
    "TVSeriesName": q_tvseries_name,
    "MovieName": q_movie_name,
    "StreamingService": q_streaming_service,
    "Recency": lambda: pick(["new movies","recent shows","latest releases","newly-released films"]),
    "Popularity": lambda: pick(["popular movies","widely-viewed movies","top charts movies","most-watched shows"]),
    "ReleaseYear": lambda: f"{pick(YEARS)} {pick(['movies','TV show'])}",
    "Decade": lambda: f"{pick(DECADES)} {pick(['movies','shows'])}",
    "FreeContent": lambda: pick(["free movies","free shows","where to watch free movies","best free TV series","free documentaries online"]),
    "AudioLanguage": q_audio_language,
    "Franchise": q_franchise,
    "Holiday": q_holiday,
    "Sport": q_sport,
    "Character": q_character,
}

# ----------------------------
# 5) Sinh câu đơn & câu kết hợp
# ----------------------------
def gen_single(persona: bool) -> str:
    e = pick(ENTITIES)
    return persona_wrap(GEN_MAP[e](), persona)

def gen_multi(persona: bool) -> str:
    # Kết hợp 2–3 thực thể thường gặp: Genre+Decade, CastAndCrew+Genre, StreamingService+Genre, ...
    combos = [
        ("Genre","Decade"),
        ("CastAndCrew","Genre"),
        ("ReleaseYear","Genre"),
        ("StreamingService","Genre"),
        ("AudioLanguage","Genre"),
        ("Franchise","ReleaseYear"),
        ("Holiday","Genre"),
        ("Sport","Genre"),
    ]
    e1,e2 = pick(combos)
    base = GEN_MAP[e1]()
    t = base
    if e2 == "Decade":
        t = re.sub(r"\bmovies\b|\bfilms\b", f"movies from the {pick(DECADES)}", base, count=1)
    elif e2 == "Genre":
        # ví dụ: "action best performances by Tom Hanks" -> đơn giản ghép prefix
        t = f"{pick(GENRES)} " + base
    elif e2 == "ReleaseYear":
        t = base + f" in {pick(YEARS)}"
    elif e2 == "StreamingService":
        t = base + f" on {pick(SERVICES)}"
    elif e2 == "AudioLanguage":
        t = f"{pick(LANGS)} " + base
    elif e2 == "Holiday":
        t = f"{pick(HOLIDAYS)} {base}"
    elif e2 == "Sport":
        t = f"{pick(SPORTS)} {base}"
    return persona_wrap(t.strip(), persona)

# ----------------------------
# 6) Hàm sinh chính
# ----------------------------
def generate(n=2500, p_multi=0.35, persona=False):
    out = []
    seen = set()
    counts = Counter()
    i = 0
    while len(out) < n and i < n*5:  # giới hạn thử 5n lần để tránh kẹt vì trùng lặp
        i += 1
        if random.random() < p_multi:
            text = gen_multi(persona)
            key = ("MULTI", text.lower())
        else:
            e = pick(ENTITIES)
            text = persona_wrap(GEN_MAP[e](), persona)
            key = (e, text.lower())
            counts[e] += 1
        # loại trùng (theo lowercase)
        if key in seen:
            continue
        seen.add(key)
        out.append({"id": f"q_{len(out)+1:06d}", "text": re.sub(r"\s+", " ", text).strip()})
    return out, counts

# ----------------------------
# 7) CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500, help="số lượng truy vấn cần sinh")
    ap.add_argument("--out", type=str, default="data/synthetic_queries.jsonl", help="đường dẫn JSONL đầu ra")
    ap.add_argument("--persona", type=int, default=1, help="0 = tắt persona flavor, 1 = bật")
    ap.add_argument("--p_multi", type=float, default=0.35, help="tỉ lệ câu kết hợp 2–3 thực thể")
    args = ap.parse_args()

    # tạo thư mục
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # sinh dữ liệu
    queries, counts = generate(n=args.n, p_multi=args.p_multi, persona=bool(args.persona))

    # lưu JSONL
    with open(args.out, "w", encoding="utf-8") as f:
        for r in queries:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # in thống kê nhanh để bạn kiểm tra độ phủ
    total_single = sum(counts.values())
    print(f"✅ Saved {len(queries)} queries to {args.out}")
    print(f"   (single-entity approx count = {total_single}, multi-entity ≈ {len(queries)-total_single})")
    print("   Per-entity single-generation counts (xấp xỉ):")
    for k,v in counts.most_common():
        print(f"   - {k:<15} {v}")

if __name__ == "__main__":
    main()
