#!/usr/bin/env bash

INPUT="$1"
i=0

amount=$(cat "$INPUT" | wc -l)

while IFS= read -r line; do
    NUM=$(echo "$line" | tr -cd '\t' | wc -c)
    if ((NUM != 8)); then
        echo "$line"
        echo "$i"
        exit 1
    fi

    if ! ((i % 1000)); then

        echo "$i / $amount"
        bc <<<"$i*100.0/$amount"

    fi
    i=$((i + 1))
done <"$INPUT"
