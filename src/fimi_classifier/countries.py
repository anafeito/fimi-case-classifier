"""Target-country extraction with a gazetteer (names, demonyms, capitals, institutions).

A country's score combines:
  * mentions in the title (x2) and summary (x1);
  * the dataset's own "countries discussed" metadata, when present (+3).

Russia is detected but excluded from *targets* by default, because EUvsDisinfo
cases come from pro-Kremlin sources: Russia is the narrator, not the target.
EU and NATO are reported separately as targeted organisations.
"""

from __future__ import annotations

import re

# ISO2: (display name, ISO3, [case-sensitive regex patterns])
GAZETTEER: dict[str, tuple[str, str, list[str]]] = {
    "UA": ("Ukraine", "UKR", [r"Ukrain\w*", r"Kyiv", r"Kiev", r"Zelensk\w+", r"Donbas+", r"Donetsk", r"Luhansk", r"Lugansk", r"Kharkiv", r"Odes+a", r"Mariupol", r"Crimea\w*", r"Bucha"]),
    "RU": ("Russia", "RUS", [r"Russia\w*", r"Moscow", r"Kremlin", r"Putin"]),
    "US": ("United States", "USA", [r"United States", r"\bUSA?\b", r"U\.S\.", r"America(?:n|ns)?\b", r"Washington", r"Pentagon", r"White House", r"Biden", r"Trump", r"\bCIA\b"]),
    "GB": ("United Kingdom", "GBR", [r"United Kingdom", r"\bUK\b", r"Britain", r"British", r"London", r"England"]),
    "DE": ("Germany", "DEU", [r"German(?:y|s)?\b", r"Berlin", r"Bundestag", r"Scholz", r"Merkel", r"Merz"]),
    "FR": ("France", "FRA", [r"France", r"French", r"Paris", r"Macron", r"Élysée|Elysee"]),
    "PL": ("Poland", "POL", [r"Poland", r"Polish", r"\bPoles?\b", r"Warsaw"]),
    "LT": ("Lithuania", "LTU", [r"Lithuania\w*", r"Vilnius"]),
    "LV": ("Latvia", "LVA", [r"Latvia\w*", r"Riga\b"]),
    "EE": ("Estonia", "EST", [r"Estonia\w*", r"Tallinn"]),
    "MD": ("Moldova", "MDA", [r"Moldova\w*", r"Chisinau", r"Chișinău", r"Transnistria\w*", r"Sandu\b", r"Gagauz\w*"]),
    "GE": ("Georgia", "GEO", [r"Georgia\b", r"Georgian\w*", r"Tbilisi", r"Abkhaz\w*", r"South Ossetia\w*"]),
    "AM": ("Armenia", "ARM", [r"Armenia\w*", r"Yerevan", r"Pashinyan"]),
    "AZ": ("Azerbaijan", "AZE", [r"Azerbaijan\w*", r"Baku"]),
    "BY": ("Belarus", "BLR", [r"Belarus\w*", r"Minsk", r"Lukashenk\w+", r"Tsikhanouskaya"]),
    "FI": ("Finland", "FIN", [r"Finland", r"Finnish", r"Helsinki", r"\bFinns?\b"]),
    "SE": ("Sweden", "SWE", [r"Sweden", r"Swedish", r"Stockholm", r"\bSwedes?\b"]),
    "NO": ("Norway", "NOR", [r"Norway", r"Norwegian", r"Oslo"]),
    "DK": ("Denmark", "DNK", [r"Denmark", r"Danish", r"Copenhagen"]),
    "CZ": ("Czechia", "CZE", [r"Czech\w*", r"Prague"]),
    "SK": ("Slovakia", "SVK", [r"Slovakia\w*", r"Slovak\b", r"Bratislava", r"Fico\b"]),
    "HU": ("Hungary", "HUN", [r"Hungar\w+", r"Budapest", r"Orb[aá]n"]),
    "RO": ("Romania", "ROU", [r"Romania\w*", r"Bucharest"]),
    "BG": ("Bulgaria", "BGR", [r"Bulgaria\w*", r"Sofia\b"]),
    "RS": ("Serbia", "SRB", [r"Serbia\w*", r"\bSerbs?\b", r"Belgrade", r"Vu[cč]i[cć]"]),
    "XK": ("Kosovo", "XKX", [r"Kosov\w+", r"Pristina"]),
    "BA": ("Bosnia and Herzegovina", "BIH", [r"Bosnia\w*", r"Sarajevo", r"Republika Srpska"]),
    "ME": ("Montenegro", "MNE", [r"Montenegr\w+", r"Podgorica"]),
    "MK": ("North Macedonia", "MKD", [r"Macedonia\w*", r"Skopje"]),
    "IT": ("Italy", "ITA", [r"Ital(?:y|ian\w*)", r"Rome\b"]),
    "ES": ("Spain", "ESP", [r"Spain", r"Spanish", r"Madrid", r"Catalonia\w*"]),
    "NL": ("Netherlands", "NLD", [r"Netherlands", r"Dutch", r"Amsterdam", r"The Hague", r"Holland"]),
    "BE": ("Belgium", "BEL", [r"Belgi(?:um|an)"]),
    "AT": ("Austria", "AUT", [r"Austria\w*", r"Vienna"]),
    "GR": ("Greece", "GRC", [r"Greece", r"Greek", r"Athens"]),
    "TR": ("Türkiye", "TUR", [r"Turkey\b", r"Türkiye", r"Turkish", r"Ankara", r"Erdo[gğ]an"]),
    "SY": ("Syria", "SYR", [r"Syria\w*", r"Damascus", r"Aleppo", r"Assad", r"Douma", r"White Helmets"]),
    "IL": ("Israel", "ISR", [r"Israel\w*", r"Tel Aviv", r"Netanyahu"]),
    "PS": ("Palestine", "PSE", [r"Palestin\w+", r"Gaza"]),
    "IR": ("Iran", "IRN", [r"Iran\w*", r"Tehran"]),
    "CN": ("China", "CHN", [r"China", r"Chinese", r"Beijing", r"\bXi Jinping\b"]),
    "KZ": ("Kazakhstan", "KAZ", [r"Kazakh\w*", r"Astana"]),
    "ML": ("Mali", "MLI", [r"\bMali\b", r"Malian", r"Bamako"]),
    "NE": ("Niger", "NER", [r"\bNiger\b", r"Nigerien", r"Niamey"]),
    "BF": ("Burkina Faso", "BFA", [r"Burkina"]),
    "CF": ("Central African Republic", "CAF", [r"Central African Republic", r"\bCAR\b"]),
    "LY": ("Libya", "LBY", [r"Libya\w*", r"Tripoli"]),
    "SD": ("Sudan", "SDN", [r"\bSudan\w*", r"Khartoum"]),
    "VE": ("Venezuela", "VEN", [r"Venezuela\w*", r"Caracas", r"Maduro"]),
    "CA": ("Canada", "CAN", [r"Canad\w+", r"Ottawa"]),
    "JP": ("Japan", "JPN", [r"Japan\w*", r"Tokyo"]),
    "AF": ("Afghanistan", "AFG", [r"Afghan\w*", r"Kabul"]),
    "IE": ("Ireland", "IRL", [r"Ireland", r"Irish"]),
    "PT": ("Portugal", "PRT", [r"Portug\w+", r"Lisbon"]),
    "HR": ("Croatia", "HRV", [r"Croatia\w*", r"Zagreb"]),
    "SI": ("Slovenia", "SVN", [r"Sloveni\w+", r"Ljubljana"]),
    "CH": ("Switzerland", "CHE", [r"Switzerland", r"Swiss", r"Geneva"]),
    "IN": ("India", "IND", [r"\bIndia\b", r"\bIndian\b", r"New Delhi"]),
    "KG": ("Kyrgyzstan", "KGZ", [r"Kyrgyz\w*", r"Kirgiz\w*", r"Bishkek"]),
    "TJ": ("Tajikistan", "TJK", [r"Tajik\w*", r"Dushanbe"]),
    "UZ": ("Uzbekistan", "UZB", [r"Uzbek\w*", r"Tashkent"]),
    "MY": ("Malaysia", "MYS", [r"Malaysia\w*", r"Kuala Lumpur"]),
    "IQ": ("Iraq", "IRQ", [r"Iraq\w*", r"Baghdad"]),
    "SA": ("Saudi Arabia", "SAU", [r"Saudi\w*", r"Riyadh"]),
    "QA": ("Qatar", "QAT", [r"Qatar\w*", r"Doha"]),
    "EG": ("Egypt", "EGY", [r"Egypt\w*", r"Cairo"]),
    "LB": ("Lebanon", "LBN", [r"Leban\w+", r"Beirut"]),
    "AL": ("Albania", "ALB", [r"Albania\w*", r"Tirana"]),
    "KP": ("North Korea", "PRK", [r"North Korea\w*", r"Pyongyang"]),
    "HK": ("Hong Kong", "HKG", [r"Hong Kong"]),
    "AU": ("Australia", "AUS", [r"Australia\w*", r"Canberra"]),
    "CU": ("Cuba", "CUB", [r"Cuba\w*", r"Havana"]),
    "BO": ("Bolivia", "BOL", [r"Bolivia\w*", r"La Paz"]),
    "MX": ("Mexico", "MEX", [r"Mexic\w+"]),
    "BR": ("Brazil", "BRA", [r"Brazil\w*", r"Brasília|Brasilia"]),
}

ORGANISATIONS: dict[str, list[str]] = {
    "EU": [r"\bEU\b", r"European Union", r"Brussels", r"European Commission", r"European Parliament", r"von der Leyen", r"Eurocrat\w*"],
    "NATO": [r"\bNATO\b", r"North Atlantic Alliance", r"Stoltenberg", r"Rutte"],
    "UN": [r"\bUN\b", r"United Nations", r"\bWHO\b", r"\bOSCE\b"],
}

# Map metadata country names (as written in the dataset) to ISO2.
NAME_TO_ISO2 = {v[0].lower(): k for k, v in GAZETTEER.items()}
NAME_TO_ISO2.update({
    "turkey": "TR", "czech republic": "CZ", "uk": "GB", "usa": "US", "us": "US", "great britain": "GB",
    "the netherlands": "NL", "moldova, republic of": "MD", "republic of moldova": "MD",
    "russian federation": "RU", "kosovo*": "XK", "macedonia": "MK", "bosnia": "BA",
    "kirgizstan": "KG", "georgian region of south ossetia": "GE",
})
# Metadata entries that expand to several countries or to an organisation.
META_GROUPS = {"baltic states": ["LT", "LV", "EE"]}
META_ORGS = {"eu": "EU", "european union": "EU", "nato": "NATO", "un": "UN"}
ISO2_TO_NAME = {k: v[0] for k, v in GAZETTEER.items()}
ISO2_TO_ISO3 = {k: v[1] for k, v in GAZETTEER.items()}

_COMPILED = {iso: [re.compile(p) for p in pats] for iso, (_, _, pats) in GAZETTEER.items()}
_ORG_COMPILED = {org: [re.compile(p) for p in pats] for org, pats in ORGANISATIONS.items()}


def _count(patterns, text: str) -> int:
    return sum(len(p.findall(text)) for p in patterns) if text else 0


def extract_countries(title: str, summary: str, metadata: list[str] | None = None,
                      exclude: tuple[str, ...] = ("RU",), min_score: float = 2.0) -> dict:
    """Return target countries ranked by score, plus targeted organisations."""
    scores: dict[str, float] = {}
    for iso, patterns in _COMPILED.items():
        s = 2.0 * _count(patterns, title) + 1.0 * _count(patterns, summary)
        if s:
            scores[iso] = s
    meta_orgs = set()
    for name in metadata or []:
        key = re.sub(r"^country:\s*", "", name.strip().lower())
        for iso in META_GROUPS.get(key, [NAME_TO_ISO2.get(key)]):
            if iso:
                scores[iso] = scores.get(iso, 0.0) + 3.0
        if key in META_ORGS:
            meta_orgs.add(META_ORGS[key])

    ranked = sorted(((s, iso) for iso, s in scores.items() if iso not in exclude), reverse=True)
    targets = [iso for s, iso in ranked if s >= min_score]
    if not targets and ranked:           # keep the single best mention as a weak target
        targets = [ranked[0][1]]
    orgs = [org for org, pats in _ORG_COMPILED.items()
            if org in meta_orgs or _count(pats, title) * 2 + _count(pats, summary) >= 1]
    return {
        "target_countries": targets,
        "target_country_names": [ISO2_TO_NAME[i] for i in targets],
        "primary_target": ISO2_TO_NAME[targets[0]] if targets else "Unspecified",
        "country_scores": {iso: s for s, iso in ranked},
        "target_orgs": orgs,
        "mentions_russia": "RU" in scores,
    }
