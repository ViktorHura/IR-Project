package main

import (
	"bufio"
	"context"
	"fmt"
	"github.com/rocketlaunchr/dataframe-go/exports"
	"golang.org/x/exp/utf8string"
	"github.com/rocketlaunchr/dataframe-go"
	"log"
	"os"
	"strings"
	"unicode"
)

func main() {

	s1 := dataframe.NewSeriesString("text", nil)
	s2 := dataframe.NewSeriesString("links", nil)
	s3 := dataframe.NewSeriesString("admiration", nil)
    s4 := dataframe.NewSeriesString("amusement", nil)
    s5 := dataframe.NewSeriesString("anger", nil)
    s6 := dataframe.NewSeriesString("annoyance", nil)
    s7 := dataframe.NewSeriesString("approval", nil)
    s8 := dataframe.NewSeriesString("caring", nil)
    s9 := dataframe.NewSeriesString("confusion", nil)
    s10 := dataframe.NewSeriesString("curiosity", nil)
    s11 := dataframe.NewSeriesString("desire", nil)
    s12 := dataframe.NewSeriesString("disappointment", nil)
    s13 := dataframe.NewSeriesString("disapproval", nil)
    s14 := dataframe.NewSeriesString("disgust", nil)
    s15 := dataframe.NewSeriesString("embarrassment", nil)
    s16 := dataframe.NewSeriesString("excitement", nil)
    s17 := dataframe.NewSeriesString("fear", nil)
    s18 := dataframe.NewSeriesString("gratitude", nil)
    s19 := dataframe.NewSeriesString("grief", nil)
    s20 := dataframe.NewSeriesString("joy", nil)
    s21 := dataframe.NewSeriesString("love", nil)
    s22 := dataframe.NewSeriesString("nervousness", nil)
    s23 := dataframe.NewSeriesString("optimism", nil)
    s24 := dataframe.NewSeriesString("pride", nil)
    s25 := dataframe.NewSeriesString("realization", nil)
    s26 := dataframe.NewSeriesString("relief", nil)
    s27 := dataframe.NewSeriesString("remorse", nil)
    s28 := dataframe.NewSeriesString("sadness", nil)
    s29 := dataframe.NewSeriesString("surprise", nil)

	df := dataframe.NewDataFrame(s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24,s25,s26,s27,s28,s29)

	fmt.Print(df.Table())

	file, err := os.Open("../../data/lexica/ImageLinksFinal.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)

	scanner.Scan()

	counter := -1
	text := "test"
	links := " "

// 	for i := 0; i < 35000; i++ {
// 		scanner.Scan()
    for scanner.Scan() {
		line := scanner.Text()
		if line[0:1] == "!" {
			if len(text) > 1 && utf8string.NewString(text).IsASCII() {
				text = strings.Map(func(r rune) rune {
					if unicode.IsPrint(r) {
						return r
					}
					return -1
				}, text)
				df.Append(nil, map[string]interface{}{
					"text":  text,
					"links": links,
					"admiration": 0,
                    "amusement": 0,
                    "anger": 0,
                    "annoyance": 0,
                    "approval": 0,
                    "caring": 0,
                    "confusion": 0,
                    "curiosity": 0,
                    "desire": 0,
                    "disappointment": 0,
                    "disapproval": 0,
                    "disgust": 0,
                    "embarrassment": 0,
                    "excitement": 0,
                    "fear": 0,
                    "gratitude": 0,
                    "grief": 0,
                    "joy": 0,
                    "love": 0,
                    "nervousness": 0,
                    "optimism": 0,
                    "pride": 0,
                    "realization": 0,
                    "relief": 0,
                    "remorse": 0,
                    "sadness": 0,
                    "surprise": 0,
				})
			}
			text = strings.TrimLeft(line, "!")
			text = strings.TrimRight(text, "\n")
			links = ""
			counter++

			if counter%5000 == 0 {
				fmt.Println(counter)
			}
		} else {
			if links != "" {
				links += ","
			}
			links += strings.TrimRight(line, "\n")
		}
	}
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}

	csvFile, err := os.Create("../../results/data/cleanData.csv")

	if err != nil {
		log.Fatalf("failed creating file: %s", err)
	}

	ctx := context.TODO()
	err = exports.ExportToCSV(ctx, csvFile, df)
	if err != nil {
		return
	}

	err = csvFile.Close()
	if err != nil {
		return
	}
}
