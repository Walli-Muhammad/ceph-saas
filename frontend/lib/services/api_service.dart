import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:image/image.dart' as img;

/// Backend URL
const String kBaseUrl = 'https://ceph-saas-production.up.railway.app';

// ── Shared exception ───────────────────────────────────────────────────────

class ApiException implements Exception {
  final int statusCode;
  final String message;
  const ApiException(this.statusCode, this.message);

  @override
  String toString() => 'ApiException($statusCode): $message';
}

// ── Image Processing ────────────────────────────────────────────────────────

Uint8List _compressImage(Uint8List imageBytes) {
  final decoded = img.decodeImage(imageBytes);
  if (decoded == null) return imageBytes;
  
  img.Image processed = decoded;
  if (decoded.width > 1024) {
    processed = img.copyResize(decoded, width: 1024);
  }
  
  return Uint8List.fromList(img.encodeJpg(processed, quality: 85));
}

// ── Diagnostic row model ───────────────────────────────────────────────────

class DiagnosticRow {
  final String parameter;
  final String value;
  final String reference;
  final String diff;
  final String comment;
  final bool isAbnormal;

  const DiagnosticRow({
    required this.parameter,
    required this.value,
    required this.reference,
    required this.diff,
    required this.comment,
    required this.isAbnormal,
  });

  factory DiagnosticRow.fromJson(Map<String, dynamic> j) => DiagnosticRow(
    parameter: j['parameter'] as String,
    value: j['value'] as String,
    reference: j['reference'] as String,
    diff: j['diff'] as String,
    comment: j['comment'] as String,
    isAbnormal: j['is_abnormal'] as bool,
  );
}

// ── Full result model ──────────────────────────────────────────────────────

class FullAnalysisResult {
  final Uint8List imageBytes;
  final List<DiagnosticRow> diagnosticsTable;
  final double pixelSize;

  /// Native-pixel coordinates for each landmark: name → (x, y)
  final String calibrationStatus;
  final String clinicalSummary;
  final Map<String, ({double x, double y})> landmarks;

  const FullAnalysisResult({
    required this.imageBytes,
    required this.diagnosticsTable,
    required this.pixelSize,
    required this.calibrationStatus,
    required this.clinicalSummary,
    required this.landmarks,
  });
}

// ── Calibration data ────────────────────────────────────────────────────────

class CalibrationData {
  final double x1, y1, x2, y2;
  final double mm;

  const CalibrationData({
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
    required this.mm,
  });
}

// ── Helpers ─────────────────────────────────────────────────────────────────

String _mimeType(String filename) {
  final ext = filename.toLowerCase();
  if (ext.endsWith('png')) return 'image/png';
  if (ext.endsWith('webp')) return 'image/webp';
  if (ext.endsWith('bmp')) return 'image/bmp';
  return 'image/jpeg';
}

List<DiagnosticRow> _parseTable(List<dynamic> list) => list
    .map(
      (m) => DiagnosticRow(
        parameter: m['parameter'] ?? '',
        value: m['value']?.toString() ?? '',
        reference: m['reference'] ?? '',
        diff: m['diff']?.toString() ?? '',
        comment: m['comment'] ?? '',
        isAbnormal: m['is_abnormal'] == true,
      ),
    )
    .toList();

Map<String, ({double x, double y})> _parseLandmarks(Map<String, dynamic> json) {
  return json.map(
    (k, v) => MapEntry(k, (
      x: (v['x'] as num).toDouble(),
      y: (v['y'] as num).toDouble(),
    )),
  );
}

FullAnalysisResult _parseResult(Map<String, dynamic> json, Uint8List imgBytes) {
  final rows = _parseTable(json['diagnostics_table'] as List);
  final pixelSize = (json['pixel_size'] as num?)?.toDouble() ?? 0.1;
  final landmarks = _parseLandmarks(
    (json['landmarks'] as Map<String, dynamic>?) ?? {},
  );
  final calibrationStatus = json['calibration_status'] as String? ?? 'unknown';
  final clinicalSummary = json['clinical_summary'] as String? ?? '';
  return FullAnalysisResult(
    imageBytes: imgBytes,
    diagnosticsTable: rows,
    pixelSize: pixelSize,
    landmarks: landmarks,
    calibrationStatus: calibrationStatus,
    clinicalSummary: clinicalSummary,
  );
}

// ── Service ────────────────────────────────────────────────────────────────

class ApiService {
  ApiService._();

  /// POST /analyze/full — initial analysis (runs PyTorch).
  /// Streams the parsed JSON immediately, then continues yielding updates
  /// as the AI clinical summary streams in.
  static Stream<FullAnalysisResult> analyzeFull({
    required Uint8List imageBytes,
    required String filename,
    CalibrationData? calibration,
  }) async* {
    final uri = Uri.parse('$kBaseUrl/analyze/full');
    final req = http.MultipartRequest('POST', uri)
      ..files.add(
        http.MultipartFile.fromBytes(
          'file',
          _compressImage(imageBytes),
          filename: filename,
          contentType: MediaType.parse(_mimeType(filename)),
        ),
      );

    if (calibration != null) {
      req.fields['calib_x1'] = calibration.x1.toString();
      req.fields['calib_y1'] = calibration.y1.toString();
      req.fields['calib_x2'] = calibration.x2.toString();
      req.fields['calib_y2'] = calibration.y2.toString();
      req.fields['calib_mm'] = calibration.mm.toString();
    }

    final streamed = await req.send().timeout(
      const Duration(minutes: 3),
      onTimeout: () => throw const ApiException(408, 'Request timed out.'),
    );

    if (streamed.statusCode != 200) {
      final body = await streamed.stream.bytesToString();
      throw ApiException(streamed.statusCode, 'Server error: $body');
    }

    final stream = streamed.stream.transform(utf8.decoder);
    
    String buffer = '';
    bool jsonParsed = false;
    FullAnalysisResult? currentResult;

    await for (final chunk in stream) {
      buffer += chunk;

      if (!jsonParsed) {
        if (buffer.contains('---END_METADATA---')) {
          final parts = buffer.split('---END_METADATA---');
          final jsonPart = parts[0];
          final textPart = parts.length > 1 ? parts.sublist(1).join('---END_METADATA---') : '';
          
          try {
            final json = jsonDecode(jsonPart) as Map<String, dynamic>;
            final imgBytes = base64Decode(json['image_base64'] as String);
            currentResult = _parseResult(json, imgBytes);
            jsonParsed = true;
            yield currentResult!;
            
            // Add remaining text to buffer to be processed below
            buffer = textPart;
          } catch (e) {
            String snippet = jsonPart.length > 100 
                ? '${jsonPart.substring(0, 50)}...${jsonPart.substring(jsonPart.length - 50)}' 
                : jsonPart;
            throw ApiException(500, 'JSON Parse Error (NEW BUILD): $e\nData Length: ${jsonPart.length}\nSnippet: $snippet');
          }
        }
      }
      
      if (jsonParsed && buffer.isNotEmpty && currentResult != null) {
        currentResult = FullAnalysisResult(
          imageBytes: currentResult!.imageBytes,
          diagnosticsTable: currentResult!.diagnosticsTable,
          pixelSize: currentResult!.pixelSize,
          calibrationStatus: currentResult!.calibrationStatus,
          clinicalSummary: currentResult!.clinicalSummary + buffer,
          landmarks: currentResult!.landmarks,
        );
        buffer = ''; // clear buffer after taking the chunk
        yield currentResult!; // Yield updated LLM streaming response
      }
    }
    
    if (currentResult == null) {
        throw const ApiException(500, 'Failed to parse JSON stream');
    }
  }

  /// POST /analyze/adjust — re-render with corrected landmarks (skips PyTorch).
  ///
  /// [landmarks] is the current mutated map from state.
  static Future<FullAnalysisResult> adjustLandmarks({
    required Uint8List imageBytes,
    required String filename,
    required Map<String, ({double x, double y})> landmarks,
  }) async {
    final uri = Uri.parse('$kBaseUrl/analyze/adjust');
    final req = http.MultipartRequest('POST', uri)
      ..files.add(
        http.MultipartFile.fromBytes(
          'file',
          imageBytes,
          filename: filename,
          contentType: MediaType.parse(_mimeType(filename)),
        ),
      );

    // Convert map of records to standard JSON map for sending
    final encodable = landmarks.map(
      (k, v) => MapEntry(k, {'x': v.x, 'y': v.y}),
    );
    req.fields['landmarks_json'] = jsonEncode(encodable);

    final streamed = await req.send().timeout(
      const Duration(seconds: 30),
      onTimeout: () =>
          throw const ApiException(408, 'Adjust request timed out.'),
    );
    final response = await http.Response.fromStream(streamed);
    if (response.statusCode != 200) {
      throw ApiException(response.statusCode, 'Adjust error: ${response.body}');
    }

    final json = jsonDecode(response.body) as Map<String, dynamic>;
    final imgBytes = base64Decode(json['image_base64'] as String);
    return _parseResult(json, imgBytes);
  }

  /// POST /analyze/chat — ask the AI a question about patient diagnostics.
  static Future<String> askQuestion({
    required List<DiagnosticRow> diagnostics,
    required String question,
  }) async {
    final uri = Uri.parse('$kBaseUrl/analyze/chat');

    // Convert diagnostics to JSON
    final diagsJson = diagnostics
        .map(
          (r) => {
            'parameter': r.parameter,
            'value': r.value,
            'reference': r.reference,
            'is_abnormal': r.isAbnormal,
          },
        )
        .toList();

    final response = await http
        .post(
          uri,
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({'diagnostics': diagsJson, 'question': question}),
        )
        .timeout(
          const Duration(minutes: 1),
          onTimeout: () => throw const ApiException(408, 'Request timed out.'),
        );

    if (response.statusCode != 200) {
      throw ApiException(response.statusCode, 'Chat error: ${response.body}');
    }

    final json = jsonDecode(response.body) as Map<String, dynamic>;
    return json['answer'] as String? ?? 'No response received.';
  }
}
