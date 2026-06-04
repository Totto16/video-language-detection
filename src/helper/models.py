from dataclasses import dataclass
from typing import Annotated, Optional, Self, override

from content.language import (
    Alpha2LanguageStr,
    Alpha3LanguageStr,
    ExactLen,
    Language,
    LongLanguageStr,
)
from helper.classifier import LinearCoeffs, MemoryPatternLinear, Model, ModelLanguage
from helper.translation import get_translator

__all__: list[str] = ["voxlingua107_ecapa_model"]

_ = get_translator()


# language_bcp47 encoding
@dataclass
class Alpha2LanguageStrRegional:
    lang: Alpha2LanguageStr
    region: str

    @staticmethod
    def __from_values_impl(
        lang: str,
        region: str,
        *,
        valid_check: bool,
    ) -> Optional["Alpha2LanguageStrRegional"]:
        lang_val: Optional[Alpha2LanguageStr] = Alpha2LanguageStr.from_str(lang)

        if lang_val is None:
            return None

        if valid_check:
            pass
            # TODO: check region str

        return Alpha2LanguageStrRegional(
            lang=lang_val,
            region=region,
        )

    @staticmethod
    def from_values(
        lang: str,
        region: str,
    ) -> Optional["Alpha2LanguageStrRegional"]:
        return Alpha2LanguageStrRegional.__from_values_impl(
            lang=lang,
            region=region,
            valid_check=True,
        )

    @staticmethod
    def from_str(inp: str) -> Optional["Alpha2LanguageStrRegional"]:
        arr: list[str] = [a.strip() for a in inp.split("-")]
        if len(arr) != 2:
            return None

        return Alpha2LanguageStrRegional.from_values(lang=arr[0], region=arr[1])

    @staticmethod
    def from_str_unsafe(inp: str) -> "Alpha2LanguageStrRegional":
        lan: Optional[Alpha2LanguageStrRegional] = Alpha2LanguageStrRegional.from_str(
            inp,
        )
        if lan is None:
            msg = _(
                "Couldn't get the Regional Language String from str: '{inp}'"  # noqa: COM812
            ).format(inp=inp)
            raise RuntimeError(msg)

        return lan


voxlingua107_ecapa_languages_count = 107


# NOTE:
# Warning: In the dataset and in the defaults of this model (see label_encoder.txt, the used ISO language code for Hebrew is obsolete (should be he instead of iw). The ISO language code for Javanese is incorrect (should be jv instead of jw). See issue #2396. (https://github.com/speechbrain/speechbrain/issues/2396)
voxlingua107_ecapa_languages: Annotated[
    list[
        tuple[
            Alpha2LanguageStr | Alpha3LanguageStr | Alpha2LanguageStrRegional,
            LongLanguageStr,
        ]
    ],
    ExactLen(voxlingua107_ecapa_languages_count),
] = [
    (
        Alpha2LanguageStr.from_str_unsafe("ab"),
        LongLanguageStr("Abkhazian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("af"),
        LongLanguageStr("Afrikaans"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("am"),
        LongLanguageStr("Amharic"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("ar"),
        LongLanguageStr("Arabic"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("as"),
        LongLanguageStr("Assamese"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("az"),
        LongLanguageStr("Azerbaijani"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("ba"),
        LongLanguageStr("Bashkir"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("be"),
        LongLanguageStr("Belarusian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("bg"),
        LongLanguageStr("Bulgarian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("bn"),
        LongLanguageStr("Bengali"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("bo"),
        LongLanguageStr("Tibetan"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("br"),
        LongLanguageStr("Breton"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("bs"),
        LongLanguageStr("Bosnian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("ca"),
        LongLanguageStr("Catalan"),
    ),
    (
        Alpha3LanguageStr.from_str_unsafe("ceb"),
        LongLanguageStr("Cebuano"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("cs"),
        LongLanguageStr("Czech"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("cy"),
        LongLanguageStr("Welsh"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("da"),
        LongLanguageStr("Danish"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("de"),
        LongLanguageStr("German"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("el"),
        LongLanguageStr("Greek"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("en"),
        LongLanguageStr("English"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("eo"),
        LongLanguageStr("Esperanto"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("es"),
        LongLanguageStr("Spanish"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("et"),
        LongLanguageStr("Estonian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("eu"),
        LongLanguageStr("Basque"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("fa"),
        LongLanguageStr("Persian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("fi"),
        LongLanguageStr("Finnish"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("fo"),
        LongLanguageStr("Faroese"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("fr"),
        LongLanguageStr("French"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("gl"),
        LongLanguageStr("Galician"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("gn"),
        LongLanguageStr("Guarani"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("gu"),
        LongLanguageStr("Gujarati"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("gv"),
        LongLanguageStr("Manx"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("ha"),
        LongLanguageStr("Hausa"),
    ),
    (
        Alpha3LanguageStr.from_str_unsafe("haw"),
        LongLanguageStr("Hawaiian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("hi"),
        LongLanguageStr("Hindi"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("hr"),
        LongLanguageStr("Croatian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("ht"),
        LongLanguageStr("Haitian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("hu"),
        LongLanguageStr("Hungarian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("hy"),
        LongLanguageStr("Armenian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("ia"),
        LongLanguageStr("Interlingua"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("id"),
        LongLanguageStr("Indonesian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("is"),
        LongLanguageStr("Icelandic"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("it"),
        LongLanguageStr("Italian"),
    ),
    (
        Alpha2LanguageStr.deserialize_str("iw"),
        LongLanguageStr("Hebrew"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("ja"),
        LongLanguageStr("Japanese"),
    ),
    (
        Alpha2LanguageStr.deserialize_str("jw"),
        LongLanguageStr("Javanese"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("ka"),
        LongLanguageStr("Georgian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("kk"),
        LongLanguageStr("Kazakh"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("km"),
        LongLanguageStr("Central Khmer"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("kn"),
        LongLanguageStr("Kannada"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("ko"),
        LongLanguageStr("Korean"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("la"),
        LongLanguageStr("Latin"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("lb"),
        LongLanguageStr("Luxembourgish"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("ln"),
        LongLanguageStr("Lingala"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("lo"),
        LongLanguageStr("Lao"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("lt"),
        LongLanguageStr("Lithuanian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("lv"),
        LongLanguageStr("Latvian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("mg"),
        LongLanguageStr("Malagasy"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("mi"),
        LongLanguageStr("Maori"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("mk"),
        LongLanguageStr("Macedonian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("ml"),
        LongLanguageStr("Malayalam"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("mn"),
        LongLanguageStr("Mongolian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("mr"),
        LongLanguageStr("Marathi"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("ms"),
        LongLanguageStr("Malay"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("mt"),
        LongLanguageStr("Maltese"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("my"),
        LongLanguageStr("Burmese"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("ne"),
        LongLanguageStr("Nepali"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("nl"),
        LongLanguageStr("Dutch"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("nn"),
        LongLanguageStr("Norwegian Nynorsk"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("no"),
        LongLanguageStr("Norwegian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("oc"),
        LongLanguageStr("Occitan"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("pa"),
        LongLanguageStr("Panjabi"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("pl"),
        LongLanguageStr("Polish"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("ps"),
        LongLanguageStr("Pushto"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("pt"),
        LongLanguageStr("Portuguese"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("ro"),
        LongLanguageStr("Romanian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("ru"),
        LongLanguageStr("Russian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("sa"),
        LongLanguageStr("Sanskrit"),
    ),
    (
        Alpha3LanguageStr.from_str_unsafe("sco"),
        LongLanguageStr("Scots"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("sd"),
        LongLanguageStr("Sindhi"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("si"),
        LongLanguageStr("Sinhala"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("sk"),
        LongLanguageStr("Slovak"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("sl"),
        LongLanguageStr("Slovenian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("sn"),
        LongLanguageStr("Shona"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("so"),
        LongLanguageStr("Somali"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("sq"),
        LongLanguageStr("Albanian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("sr"),
        LongLanguageStr("Serbian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("su"),
        LongLanguageStr("Sundanese"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("sv"),
        LongLanguageStr("Swedish"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("sw"),
        LongLanguageStr("Swahili"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("ta"),
        LongLanguageStr("Tamil"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("te"),
        LongLanguageStr("Telugu"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("tg"),
        LongLanguageStr("Tajik"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("th"),
        LongLanguageStr("Thai"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("tk"),
        LongLanguageStr("Turkmen"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("tl"),
        LongLanguageStr("Tagalog"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("tr"),
        LongLanguageStr("Turkish"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("tt"),
        LongLanguageStr("Tatar"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("uk"),
        LongLanguageStr("Ukrainian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("ur"),
        LongLanguageStr("Urdu"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("uz"),
        LongLanguageStr("Uzbek"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("vi"),
        LongLanguageStr("Vietnamese"),
    ),
    (
        Alpha3LanguageStr.from_str_unsafe("war"),
        LongLanguageStr("Waray"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("yi"),
        LongLanguageStr("Yiddish"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("yo"),
        LongLanguageStr("Yoruba"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("zh"),
        LongLanguageStr("Mandarin Chinese"),
    ),
]


if len(voxlingua107_ecapa_languages) != voxlingua107_ecapa_languages_count:
    raise RuntimeError("UNREACHABLE")  # noqa: EM101


class ModelLanguageForList(ModelLanguage):
    __languages: list[
        tuple[
            Alpha2LanguageStr | Alpha3LanguageStr | Alpha2LanguageStrRegional,
            LongLanguageStr,
        ]
    ]

    def __init__(
        self: Self,
        languages: list[
            tuple[
                Alpha2LanguageStr | Alpha3LanguageStr | Alpha2LanguageStrRegional,
                LongLanguageStr,
            ]
        ],
    ) -> None:
        super().__init__()
        self.__languages = languages

    @override
    def is_valid_language(self: Self, language: Language) -> Optional[str]:
        for short_str, long_str in self.__languages:
            if short_str == language.short:
                if language.long != long_str:
                    return _("Long language doesn't match")
                return None

        return _("This dataset has no such language")


voxlingua107_ecapa_model: Model = Model(
    name="voxlingua107",
    sample_count=voxlingua107_ecapa_languages_count,
    source="speechbrain/lang-id-voxlingua107-ecapa",
    bitrate=16000,
    model_language=ModelLanguageForList(voxlingua107_ecapa_languages),
    memory_pattern=MemoryPatternLinear(
        coeffs=LinearCoeffs(c=121287679.99999952, m=12845499.313230773),
    ),
)


commonlanguage_ecapa_languages_count = 45


commonlanguage_ecapa_languages: Annotated[
    list[
        tuple[
            Alpha2LanguageStr | Alpha3LanguageStr | Alpha2LanguageStrRegional,
            LongLanguageStr,
        ]
    ],
    ExactLen(commonlanguage_ecapa_languages_count),
] = [
    (
        Alpha2LanguageStr.from_str_unsafe("ar"),
        LongLanguageStr("Arabic"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("eu"),
        LongLanguageStr("Basque"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("br"),
        LongLanguageStr("Breton"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("ca"),
        LongLanguageStr("Catalan"),
    ),
    (
        Alpha2LanguageStrRegional.from_str_unsafe("zh-CH"),
        LongLanguageStr("Chinese_China"),
    ),
    (
        Alpha2LanguageStrRegional.from_str_unsafe("zh-HK"),
        LongLanguageStr("Chinese_Hongkong"),
    ),
    (
        Alpha2LanguageStrRegional.from_str_unsafe("zh-TW"),
        LongLanguageStr("Chinese_Taiwan"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("cv"),
        LongLanguageStr("Chuvash"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("cs"),
        LongLanguageStr("Czech"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("dv"),
        LongLanguageStr("Dhivehi"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("nl"),
        LongLanguageStr("Dutch"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("en"),
        LongLanguageStr("English"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("eo"),
        LongLanguageStr("Esperanto"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("et"),
        LongLanguageStr("Estonian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("fr"),
        LongLanguageStr("French"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("fy"),
        LongLanguageStr("Frisian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("ka"),
        LongLanguageStr("Georgian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("de"),
        LongLanguageStr("German"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("el"),
        LongLanguageStr("Greek"),
    ),
    (
        Alpha3LanguageStr.from_str_unsafe("cnh"),
        LongLanguageStr("Hakha_Chin"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("id"),
        LongLanguageStr("Indonesian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("ia"),
        LongLanguageStr("Interlingua"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("it"),
        LongLanguageStr("Italian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("ja"),
        LongLanguageStr("Japanese"),
    ),
    (
        Alpha3LanguageStr.from_str_unsafe("kab"),
        LongLanguageStr("Kabyle"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("rw"),
        LongLanguageStr("Kinyarwanda"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("ky"),
        LongLanguageStr("Kyrgyz"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("lv"),
        LongLanguageStr("Latvian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("mt"),
        LongLanguageStr("Maltese"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("mn"),
        LongLanguageStr("Mongolian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("fa"),
        LongLanguageStr("Persian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("pl"),
        LongLanguageStr("Polish"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("pt"),
        LongLanguageStr("Portuguese"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("ro"),
        LongLanguageStr("Romanian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("rm"),
        LongLanguageStr("Romansh_Sursilvan"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("ru"),
        LongLanguageStr("Russian"),
    ),
    (
        Alpha3LanguageStr.from_str_unsafe("sah"),
        LongLanguageStr("Sakha"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("sl"),
        LongLanguageStr("Slovenian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("es"),
        LongLanguageStr("Spanish"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("sv"),
        LongLanguageStr("Swedish"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("ta"),
        LongLanguageStr("Tamil"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("tt"),
        LongLanguageStr("Tatar"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("tr"),
        LongLanguageStr("Turkish"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("uk"),
        LongLanguageStr("Ukrainian"),
    ),
    (
        Alpha2LanguageStr.from_str_unsafe("cy"),
        LongLanguageStr("Welsh"),
    ),
]


if len(commonlanguage_ecapa_languages) != commonlanguage_ecapa_languages_count:
    raise RuntimeError("UNREACHABLE")  # noqa: EM101


commonlanguage_ecapa_model: Model = Model(
    name="commonlanguage",
    sample_count=commonlanguage_ecapa_languages_count,
    source="speechbrain/lang-id-commonlanguage_ecapa",
    bitrate=16000,
    model_language=ModelLanguageForList(commonlanguage_ecapa_languages),
    memory_pattern=None,  # TODO
)
