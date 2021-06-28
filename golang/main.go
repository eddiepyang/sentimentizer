package main

import (
	"archive/zip"
	"bufio"
	"fmt"
	"log"
	"os"
	"yelp_nlp/golang/parser"
)

func main() {

	infile, err := zip.OpenReader("../data/archive.zip")
	if err != nil {
		fmt.Println(err)
		log.Fatal(err)
	}

	// for _, f := range infile.File {
	// 	fmt.Printf("%s \n", f.Name)
	// }

	defer infile.Close()
	file, err := infile.Open("yelp_academic_dataset_review.json")
	if err != nil {

		log.Fatal(err)

	}

	iterator := bufio.NewScanner(file)
	// var line int
	// var tmpline parser.Jsonline
	// jsonDump := make(map[int]parser.Jsonline)

	// for iterator.Scan() {

	// 	err := json.Unmarshal(iterator.Bytes(), &tmpline)
	// 	if err != nil {
	// 		log.Fatal(err)
	// 	}

	// 	jsonDump[line] = tmpline
	// 	if line == 5 {
	// 		break
	// 	}
	// 	//fmt.Println(line, tmpline)
	// 	line += 1
	// }

	// //outf, err := os.OpenFile("../data/test.jsonl", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	//outf, err := os.OpenFile("../data/test.jsonl", os.O_CREATE|os.O_WRONLY, 0666)
	if err != nil {
		//fmt.Println(err)
		log.Fatal(err)
	}

	// writing to parquet
	parser.ParquetWrite(iterator)

	// n := parser.WriteFile(*outf, jsonDump)
	// outf.Sync()
	// outf.Close()

	// // validates map exists
	// outfile := parser.ReadJSON("../data/test.jsonl",
	// 	func(dc map[string]interface{}) bool {
	// 		if dc != nil {
	// 			return true
	// 		}
	// 		return false
	// 	},
	// )

	// fmt.Println(n, len(outfile), "\n\n\n")
	// fmt.Println(outfile)
	// for k, v := range outfile {
	// 	fmt.Printf("Key: %v \n", k)
	// 	fmt.Printf("The following lines were read: \n %v \n`", v)
	// }

	// if err != nil {
	// 	log.Fatal(err)
	// }
	os.Exit(0)

}
