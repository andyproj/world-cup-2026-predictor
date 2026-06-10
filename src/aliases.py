"""Team-name normalization used across data sources."""

ALIASES = {
    "United States": "USA",
    "United States of America": "USA",
    "USMNT": "USA",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Türkiye": "Turkey",
    "Bosnia-Herzegovina": "Bosnia & Herzegovina",
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "Czechia": "Czech Republic",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Democratic Republic of the Congo": "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Cape Verde Islands": "Cape Verde",
    "Cabo Verde": "Cape Verde",
    "Curacao": "Curaçao",
    "China PR": "China",
    "IR Iran": "Iran",
    "Chinese Taipei": "Taiwan",
    "Macedonia": "North Macedonia",
    "Swaziland": "Eswatini",
}


def canonical_team(name: object) -> str:
    """Return a consistent team name while safely handling null values."""
    if name is None:
        return ""
    value = str(name).strip()
    return ALIASES.get(value, value)
