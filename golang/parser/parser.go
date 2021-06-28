package parser

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"

	"github.com/xitongsys/parquet-go-source/local"
	"github.com/xitongsys/parquet-go/parquet"
	"github.com/xitongsys/parquet-go/writer"
)

type Jsonline interface{}

func ShowFiles(path string) error {
	files, err := ioutil.ReadDir(path)
	if err != nil {
		log.Fatal(err)
		return err
	}

	for _, item := range files {
		fmt.Println(item.Name(), item.IsDir())
	}
	return nil
}

// func WriteDataToFileAsJSON(data interface{}, filedir string) (int, error) {
// 	//write data as buffer to json encoder
// 	buffer := new(bytes.Buffer)
// 	encoder := json.NewEncoder(buffer)
// 	encoder.SetIndent("", "\t")

// 	err := encoder.Encode(data)
// 	if err != nil {
// 		return 0, err
// 	}
// 	file, err := os.OpenFile(filedir, os.O_RDWR|os.O_CREATE, 0755)
// 	if err != nil {
// 		return 0, err
// 	}
// 	n, err := file.Write(buffer.Bytes())
// 	if err != nil {
// 		return 0, err
// 	}
// 	return n, nil
// }

func WriteFile(outfile os.File, data map[int]Jsonline) int {
	n := 0
	datawriter := bufio.NewWriter(&outfile)

	for _, item := range data {
		bytesj, err := json.Marshal(item)
		if err != nil {
			log.Fatal(err)
		}
		// fmt.Println("writing: \n\n", item)
		i, err := datawriter.Write(bytesj)
		if err != nil {
			fmt.Println("err occurred")
			log.Fatal(err)
		}
		n += i

	}
	datawriter.Flush()
	return n
}

func ReadJSON(fileName string, filter func(map[string]interface{}) bool) []map[string]interface{} {
	file, _ := os.Open(fileName)
	defer file.Close()

	decoder := json.NewDecoder(file)

	filteredData := []map[string]interface{}{}

	// Read the array open bracket
	//decoder.Token()

	var data map[string]interface{}

	for decoder.More() {
		decoder.Decode(&data)

		if filter(data) {
			filteredData = append(filteredData, data)
		}
	}

	return filteredData
}

func ParquetWrite(scanner *bufio.Scanner) {

	writefile, err := local.NewLocalFileWriter("../data/reviews.parquet")
	if err != nil {
		log.Fatal(err)
	}
	defer writefile.Close()

	type Schema struct {
		Business_id string  `parquet:"name=name, type=BYTE_ARRAY"`
		Cool        float64 `parquet:"name=cool, type=DOUBLE"`
		Funny       float64 `parquet:"name=funny, type=DOUBLE"`
		Review_id   string  `parquet:"name=review_id, type=BYTE_ARRAY"`
		Stars       float64 `parquet:"name=stars, type=DOUBLE"`
		Text        string  `parquet:"name=text, type=BYTE_ARRAY"`
		Useful      float64 `parquet:"name=useful, type=DOUBLE"`
		User_id     string  `parquet:"name=user_id, type=BYTE_ARRAY"`
		Date        string  `parquet:"name=date, type=BYTE_ARRAY"`
		Ignored     int32   //without parquet tag and won't write
	}

	var schema Schema = Schema{}

	//datawriter := bufio.NewWriter(&outfile)
	//var tmpline Jsonline

	pw, err := writer.NewParquetWriter(writefile, new(Schema), 1)
	if err != nil {
		log.Fatal(err)
	}
	pw.RowGroupSize = 128 * 1024 * 1024 //128M
	pw.PageSize = 8 * 1024 * 1024       //8M
	pw.CompressionType = parquet.CompressionCodec_SNAPPY
	line := 0

	for scanner.Scan() {
		if line < 2 {
			err := json.Unmarshal(scanner.Bytes(), &schema)
			if err != nil {
				log.Fatal(err)
			}
			// values := tmpline.(string)
			// fmt.Println(values)
			// data := Schema{
			// 	Business_id: values,
			// }
			err = pw.Write(schema)
			if err != nil {
				log.Fatal(err)
			}
			line += 1
		} else {
			break
		}

	}
	if err = pw.WriteStop(); err != nil {
		log.Println("WriteStop error", err)
		return
	}

}
