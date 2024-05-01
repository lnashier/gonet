package mnist

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
)

func readData(imagesPath, labelsPath string) ([][][]uint8, []uint8, error) {
	images, err := readImages(imagesPath)
	if err != nil {
		return nil, nil, err
	}

	labels, err := readLabels(labelsPath)
	if err != nil {
		return nil, nil, err
	}

	return images, labels, nil
}

func normalizeData(outputSize int, images [][][]uint8, labels []uint8) ([][]float64, [][]float64, error) {
	var inputs [][]float64
	var outputs [][]float64
	for i, img := range images {
		inputs = append(inputs, flattenImage(img))
		output := make([]float64, outputSize)
		output[labels[i]] = 1.0 // One-hot encode the label
		outputs = append(outputs, output)
	}
	return inputs, outputs, nil
}

func readLabels(filename string) ([]uint8, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	gzipReader, err := gzip.NewReader(file)
	if err != nil {
		return nil, err
	}
	defer gzipReader.Close()

	var magic uint32
	if err := binary.Read(gzipReader, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != 0x0801 {
		return nil, fmt.Errorf("invalid magic number for label file")
	}

	var numLabels uint32
	if err := binary.Read(gzipReader, binary.BigEndian, &numLabels); err != nil {
		return nil, err
	}

	labels := make([]uint8, numLabels)
	if err := binary.Read(gzipReader, binary.BigEndian, &labels); err != nil {
		return nil, err
	}

	return labels, nil
}

func readImages(filename string) ([][][]uint8, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	gzipReader, err := gzip.NewReader(file)
	if err != nil {
		return nil, err
	}
	defer gzipReader.Close()

	var magic uint32
	if err := binary.Read(gzipReader, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != 0x0803 {
		return nil, fmt.Errorf("invalid magic number for image file")
	}

	var numImages uint32
	if err := binary.Read(gzipReader, binary.BigEndian, &numImages); err != nil {
		return nil, err
	}

	var numRows, numCols uint32
	if err := binary.Read(gzipReader, binary.BigEndian, &numRows); err != nil {
		return nil, err
	}
	if err := binary.Read(gzipReader, binary.BigEndian, &numCols); err != nil {
		return nil, err
	}

	images := make([][][]uint8, numImages)

	for i := range images {
		images[i] = make([][]uint8, numRows)
		for j := range images[i] {
			images[i][j] = make([]uint8, numCols)
			for k := range images[i][j] {
				var pixel uint8
				err = binary.Read(gzipReader, binary.BigEndian, &pixel)
				if err != nil {
					return nil, err
				}
				images[i][j][k] = pixel
			}
		}
	}

	return images, nil
}

func saveImage(imageData [][]uint8, filename string) error {
	numRows := len(imageData)
	numCols := len(imageData[0])

	img := image.NewGray(image.Rect(0, 0, numCols, numRows))

	for i := 0; i < numRows; i++ {
		for j := 0; j < numCols; j++ {
			img.SetGray(j, i, color.Gray{Y: imageData[i][j]})
		}
	}

	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	err = png.Encode(file, img)
	if err != nil {
		return err
	}

	return nil
}

func flattenImage(image [][]uint8) []float64 {
	var flattened []float64
	for _, row := range image {
		for _, col := range row {
			flattened = append(flattened, float64(col)/255.0)
		}
	}
	return flattened
}
