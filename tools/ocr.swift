import Foundation
import Vision
import ImageIO
import CoreImage
import PDFKit

// ---------- options ----------
var jsonOutput = false
var enhance = false
var correction = true
var langs = ["zh-Hans", "en-US"]
var files: [String] = []

let argv = CommandLine.arguments
var idx = 1
while idx < argv.count {
    switch argv[idx] {
    case "--json": jsonOutput = true
    case "--enhance": enhance = true
    case "--no-correction": correction = false
    case "--lang":
        idx += 1
        if idx < argv.count {
            switch argv[idx] {
            case "zh": langs = ["zh-Hans"]
            case "en": langs = ["en-US"]
            default: langs = ["zh-Hans", "en-US"]
            }
        }
    default: files.append(argv[idx])
    }
    idx += 1
}

guard !files.isEmpty else {
    FileHandle.standardError.write(Data(
        "usage: ocrfast <file> [file ...] [--json] [--lang zh|en|mixed] [--enhance] [--no-correction]\n".utf8))
    exit(1)
}

// ---------- preprocessing ----------
func prepare(_ cg: CGImage) -> CGImage {
    let ctx = CIContext(options: [.cacheIntermediates: false])
    var img = CIImage(cgImage: cg)
    let shortSide = min(cg.width, cg.height)
    if shortSide < 1400 {
        let scale = max(2.0, Double(max(cg.width, cg.height)) / 1400.0)
        if let f = CIFilter(name: "CILanczosScaleTransform") {
            f.setValue(img, forKey: kCIInputImageKey)
            f.setValue(scale, forKey: kCIInputScaleKey)
            f.setValue(1.0, forKey: kCIInputAspectRatioKey)
            if let out = f.outputImage { img = out }
        }
    }
    if enhance {
        if let f = CIFilter(name: "CIColorControls") {
            f.setValue(img, forKey: kCIInputImageKey)
            f.setValue(1.15, forKey: kCIInputContrastKey)
            f.setValue(0.0, forKey: kCIInputSaturationKey)
            if let out = f.outputImage { img = out }
        }
    }
    return ctx.createCGImage(img, from: img.extent.integral) ?? cg
}

// ---------- recognition ----------
struct OCRLine: Codable {
    let text: String
    let confidence: Double
    let box: [Double]
}

func recognize(_ cg: CGImage) -> ([OCRLine], String?) {
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    req.usesLanguageCorrection = correction
    req.recognitionLanguages = langs
    let handler = VNImageRequestHandler(cgImage: cg, options: [:])
    do { try handler.perform([req]) } catch { return ([], "\(error)") }
    var out: [OCRLine] = []
    for o in req.results ?? [] {
        if let c = o.topCandidates(1).first {
            let b = o.boundingBox
            out.append(OCRLine(text: c.string,
                               confidence: Double(c.confidence),
                               box: [Double(b.origin.x), Double(b.origin.y), Double(b.size.width), Double(b.size.height)]))
        }
    }
    return (out, nil)
}

func ocrImage(_ cg: CGImage) -> ([OCRLine], String?) {
    recognize(prepare(cg))
}

func ocrPDF(_ url: URL) -> ([[OCRLine]], String?) {
    guard let doc = PDFDocument(url: url) else {
        return ([], "cannot open pdf: \(url.path)")
    }
    var pages: [[OCRLine]] = []
    for index in 0..<doc.pageCount {
        guard let page = doc.page(at: index) else { continue }
        let embedded = (page.string ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        var lines: [OCRLine] = []
        if !embedded.isEmpty {
            lines = [OCRLine(text: embedded, confidence: 1.0, box: [0, 0, 1, 1])]
        } else {
            let bounds = page.bounds(for: .mediaBox)
            let scale: CGFloat = 200.0 / 72.0
            let size = CGSize(width: bounds.width * scale, height: bounds.height * scale)
            let thumb = page.thumbnail(of: size, for: .mediaBox)
            guard let cg = thumb.cgImage(forProposedRect: nil, context: nil, hints: nil) else { continue }
            let (r, _) = ocrImage(cg)
            lines = r
        }
        pages.append(lines)
    }
    return (pages, nil)
}

// ---------- main ----------
struct PageResult: Codable { let page: Int; let lines: [OCRLine] }
struct FileResult: Codable { let file: String; let pages: [PageResult] }

var results: [FileResult] = []
var hadError = false

for path in files {
    let url = URL(fileURLWithPath: path)
    var pages: [[OCRLine]] = []
    if url.pathExtension.lowercased() == "pdf" {
        let (p, err) = ocrPDF(url)
        if let err = err {
            FileHandle.standardError.write(Data("\(err)\n".utf8))
            hadError = true
            continue
        }
        pages = p
    } else {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            FileHandle.standardError.write(Data("cannot load image: \(path)\n".utf8))
            hadError = true
            continue
        }
        let (lines, err) = ocrImage(cg)
        if let err = err {
            FileHandle.standardError.write(Data("vision error: \(err)\n".utf8))
            hadError = true
            continue
        }
        pages = [lines]
    }
    results.append(FileResult(file: path, pages: pages.enumerated().map { PageResult(page: $0.offset + 1, lines: $0.element) }))
}

if jsonOutput {
    let data = try! JSONEncoder().encode(results)
    print(String(data: data, encoding: .utf8)!)
} else {
    for file in results {
        if results.count > 1 { print("--- \(file.file) ---") }
        for page in file.pages {
            if file.pages.count > 1 { print("--- Page \(page.page) ---") }
            for line in page.lines { print(line.text) }
        }
    }
}
exit(hadError ? 2 : 0)
