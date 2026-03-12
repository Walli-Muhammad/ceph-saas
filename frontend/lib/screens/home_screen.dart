import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../services/api_service.dart';
import '../services/pdf_service.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  // ── Analysis state ─────────────────────────────────────────────────────────
  Uint8List? _selectedBytes;
  String? _selectedFilename;
  FullAnalysisResult? _result;
  bool _isLoading = false;
  bool _isAdjusting = false;
  String? _errorText;

  /// Current landmark coordinates in native image pixels — mutated on drag.
  Map<String, ({double x, double y})>? _landmarks;

  // ── AI Chatbot state ───────────────────────────────────────────────────────
  String _chatQuestion = '';
  String? _chatAnswer;
  bool _isChatLoading = false;
  final TextEditingController _chatController = TextEditingController();

  // ── Calibration state ──────────────────────────────────────────
  bool _isCalibrating = false;
  Offset? _calPoint1;
  Offset? _calPoint2;
  Offset? _calRel1;
  Offset? _calRel2;
  double? _calMm;
  CalibrationData? _calibration;

  // Native dimensions decoded once per selected image
  int? _nativeW;
  int? _nativeH;
  // Key to measure the image widget render size
  final _imgKey = GlobalKey();
  // Key used by onPanUpdate to convert global pointer position → Stack-local coords
  final _annotatedStackKey = GlobalKey();

  // ── Loupe (Precision Zoom) state ───────────────────────────────
  bool _isDragging = false;
  Offset _dragScreenPos = Offset.zero;
  String _activeLandmarkName = '';

  // ── Pick image ────────────────────────────────────────────────────────────
  Future<void> _pickImage() async {
    final picked = await FilePicker.platform.pickFiles(
      type: FileType.image,
      withData: true,
    );
    if (picked == null || picked.files.isEmpty) return;
    final f = picked.files.first;
    setState(() {
      _selectedBytes = f.bytes;
      _selectedFilename = f.name;
      _result = _errorText = null;
      _nativeW = _nativeH = null;
      _chatQuestion = '';
      _chatAnswer = null;
      _isChatLoading = false;
      _chatController.clear();
    });
    // Decode native image dimensions for accurate calibration mapping
    if (f.bytes != null) {
      final codec = await ui.instantiateImageCodec(f.bytes!);
      final frame = await codec.getNextFrame();
      if (mounted) {
        setState(() {
          _nativeW = frame.image.width;
          _nativeH = frame.image.height;
        });
      }
      frame.image.dispose();
    }
  }

  // ── Calibration interaction ────────────────────────────────────────────────
  void _resetCalibration() {
    setState(() {
      _isCalibrating = false;
      _calPoint1 = _calPoint2 = null;
      _calRel1 = _calRel2 = null;
      _calMm = null;
      _calibration = null;
    });
  }

  void _onCalibrationTap(TapDownDetails d) {
    if (!_isCalibrating) return;
    setState(() {
      if (_calPoint1 == null) {
        _calPoint1 = d.localPosition;
      } else if (_calPoint2 == null) {
        _calPoint2 = d.localPosition;
        _showMmDialog();
      }
    });
  }

  Offset? _screenToRelative(Offset screenPos) {
    if (_nativeW == null || _nativeH == null) return null;
    final RenderBox? box =
        _imgKey.currentContext?.findRenderObject() as RenderBox?;
    if (box == null) return null;

    final ww = box.size.width;
    final wh = box.size.height;
    final imageAR = _nativeW! / _nativeH!;
    final widgetAR = ww / wh;

    double renderedW, renderedH, dx, dy;
    if (widgetAR > imageAR) {
      renderedH = wh;
      renderedW = wh * imageAR;
      dx = (ww - renderedW) / 2.0;
      dy = 0.0;
    } else {
      renderedW = ww;
      renderedH = ww / imageAR;
      dx = 0.0;
      dy = (wh - renderedH) / 2.0;
    }

    // Is the tap outside the actual image box?
    if (screenPos.dx < dx ||
        screenPos.dx > dx + renderedW ||
        screenPos.dy < dy ||
        screenPos.dy > dy + renderedH) {
      return null;
    }

    final relX = (screenPos.dx - dx) / renderedW;
    final relY = (screenPos.dy - dy) / renderedH;
    return Offset(relX.clamp(0.0, 1.0), relY.clamp(0.0, 1.0));
  }

  void _showMmDialog() {
    final rel1 = _screenToRelative(_calPoint1!);
    final rel2 = _screenToRelative(_calPoint2!);

    if (rel1 == null || rel2 == null) {
      setState(
        () => _errorText = 'Calibration points must be inside the image.',
      );
      _resetCalibration();
      return;
    }

    _calRel1 = rel1;
    _calRel2 = rel2;

    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (ctx) {
        final tc = TextEditingController(text: '20.0');
        return AlertDialog(
          backgroundColor: _card,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          title: const Text(
            'Enter Real Distance',
            style: TextStyle(color: Colors.white, fontSize: 18),
          ),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text(
                'What is the actual distance (in millimeters) between these two points?',
                style: TextStyle(color: Colors.white70, fontSize: 14),
              ),
              const SizedBox(height: 16),
              TextField(
                controller: tc,
                keyboardType: const TextInputType.numberWithOptions(
                  decimal: true,
                ),
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.center,
                decoration: InputDecoration(
                  suffixText: 'mm',
                  suffixStyle: const TextStyle(
                    color: Colors.white54,
                    fontSize: 18,
                  ),
                  filled: true,
                  fillColor: _dark,
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                    borderSide: BorderSide.none,
                  ),
                ),
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.pop(ctx);
                _resetCalibration();
              },
              child: const Text(
                'Cancel',
                style: TextStyle(color: Colors.white54),
              ),
            ),
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: _teal,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
              onPressed: () {
                final val = double.tryParse(tc.text.trim());
                if (val != null && val > 0) {
                  setState(() {
                    _calMm = val;
                    _calibration = CalibrationData(
                      x1: _calRel1!.dx,
                      y1: _calRel1!.dy,
                      x2: _calRel2!.dx,
                      y2: _calRel2!.dy,
                      mm: val,
                    );
                    _isCalibrating = false;
                  });
                  Navigator.pop(ctx);
                }
              },
              child: const Text(
                'Confirm',
                style: TextStyle(color: Colors.white),
              ),
            ),
          ],
        );
      },
    );
  }

  // ── Design tokens ──────────────────────────────────────────────────────────
  static const _dark = Color(0xFF1C1C2E);
  static const _card = Color(0xFF252540);
  static const _panel = Color(0xFF1E1E35);
  static const _blue = Color(0xFF1A73E8);
  static const _teal = Color(0xFF00BFA5);
  static const _border = Color(0xFF2E2E50);
  static const _normal = Color(0xFF00BFA5);
  static const _warn = Color(0xFFFF6B35);
  static const _green = Color(0xFF76FF03); // ruler line color

  // ── Analyze ───────────────────────────────────────────────────────────────
  Future<void> _analyze() async {
    if (_selectedBytes == null) return;
    setState(() {
      _isLoading = true;
      _errorText = null;
      _result = null;
      _landmarks = null;
      _chatQuestion = '';
      _chatAnswer = null;
      _isChatLoading = false;
      _chatController.clear();
    });
    
    try {
      final stream = ApiService.analyzeFull(
        imageBytes: _selectedBytes!,
        filename: _selectedFilename ?? 'image.jpg',
        calibration: _calibration,
      );
      
      await for (final res in stream) {
        if (!mounted) break;
        setState(() {
          _result = res;
          _landmarks = Map.from(res.landmarks);
          _isLoading = false; // Hide loading spinner as soon as first chunk arrives
        });
      }
    } on ApiException catch (e) {
      if (mounted) setState(() => _errorText = e.message);
    } catch (e) {
      if (mounted) setState(() => _errorText = 'Unexpected error: $e');
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  // ── Adjust landmarks (re-render without PyTorch) ──────────────────────────
  Future<void> _adjustLandmarks() async {
    if (_selectedBytes == null || _landmarks == null || _result == null) return;
    setState(() => _isAdjusting = true);
    try {
      final res = await ApiService.adjustLandmarks(
        imageBytes: _selectedBytes!,
        filename: _selectedFilename ?? 'image.jpg',
        landmarks: _landmarks!,
      );
      setState(() {
        _result = res;
        _landmarks = Map.from(res.landmarks);
      });
    } on ApiException catch (e) {
      setState(() => _errorText = e.message);
    } catch (e) {
      setState(() => _errorText = 'Unexpected error: $e');
    } finally {
      setState(() => _isAdjusting = false);
    }
  }

  void _reset() => setState(() {
    _selectedBytes = _selectedFilename = _result = _errorText = null;
    _isLoading = _isAdjusting = false;
    _nativeW = _nativeH = null;
    _landmarks = null;
  });

  Future<void> _exportPdf() async {
    if (_result == null) return;
    await generateClinicalReport(
      imageBytes: _result!.imageBytes,
      diagnostics: _result!.diagnosticsTable,
      clinicalSummary: _result!.clinicalSummary,
    );
  }

  // ── Build ──────────────────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) => Scaffold(
    backgroundColor: _dark,
    appBar: _appBar(),
    body: SafeArea(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(20, 16, 20, 16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _header(),
            const SizedBox(height: 12),
            Expanded(child: _body()),
            const SizedBox(height: 14),
            if (_errorText != null) _errorBanner(),
            if (!_isLoading) _actionButtons(),
          ],
        ),
      ),
    ),
  );

  // ── Central body ──────────────────────────────────────────────────────────
  Widget _body() {
    if (_isLoading) return _loadingPanel();
    if (_result != null) return _resultLayout(_result!);
    if (_selectedBytes != null) return _previewWithCalibration();
    return _emptyPanel();
  }

  // ── Preview with calibration overlay ─────────────────────────────────────
  Widget _previewWithCalibration() {
    return Stack(
      children: [
        // The X-ray preview
        GestureDetector(
          key: _imgKey,
          onTapDown: _onCalibrationTap,
          child: ClipRRect(
            borderRadius: BorderRadius.circular(16),
            child: Stack(
              fit: StackFit.expand,
              children: [
                Image.memory(
                  _selectedBytes!,
                  fit: BoxFit.contain,
                  width: double.infinity,
                  height: double.infinity,
                ),
                // The Ruler custom paint layer
                if (_isCalibrating || _calibration != null)
                  CustomPaint(
                    painter: _RulerPainter(_calPoint1, _calPoint2, _green),
                    child: Container(),
                  ),
                // Badge
                Positioned(top: 10, right: 10, child: _chip('Selected', _blue)),
              ],
            ),
          ),
        ),
        // Calibration Help Text & Cancel Button
        if (_isCalibrating)
          Positioned(
            top: 16,
            left: 16,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              decoration: BoxDecoration(
                color: _dark.withOpacity(0.9),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: _green.withOpacity(0.5)),
              ),
              child: Row(
                children: [
                  const Icon(Icons.ads_click, color: _green, size: 20),
                  const SizedBox(width: 10),
                  Text(
                    _calPoint1 == null
                        ? 'Tap first point of ruler'
                        : 'Tap second point of ruler',
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(width: 16),
                  InkWell(
                    onTap: _resetCalibration,
                    child: const Icon(
                      Icons.close,
                      color: Colors.white54,
                      size: 20,
                    ),
                  ),
                ],
              ),
            ),
          ),
        // Confirmed Badge
        if (_calibration != null && !_isCalibrating)
          Positioned(
            top: 16,
            left: 16,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: _green.withOpacity(0.15),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: _green.withOpacity(0.4)),
              ),
              child: Row(
                children: [
                  const Icon(Icons.check_circle, color: _green, size: 16),
                  const SizedBox(width: 8),
                  Text(
                    '${_calMm}mm set',
                    style: const TextStyle(
                      color: _green,
                      fontSize: 13,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(width: 12),
                  InkWell(
                    onTap: _resetCalibration,
                    child: const Icon(Icons.close, color: _green, size: 16),
                  ),
                ],
              ),
            ),
          ),
        // Hovering "Calibrate Scale" Action Button Button
        if (!_isCalibrating && _calibration == null)
          Positioned(
            bottom: 20,
            left: 20,
            right: 20,
            child: ElevatedButton.icon(
              style: ElevatedButton.styleFrom(
                backgroundColor: _dark.withOpacity(0.85),
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 14),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                  side: const BorderSide(color: Colors.white24),
                ),
              ),
              icon: const Icon(Icons.straighten, size: 20),
              label: const Text(
                'Calibrate Scale (Optional)',
                style: TextStyle(fontSize: 15, fontWeight: FontWeight.w600),
              ),
              onPressed: () => setState(() => _isCalibrating = true),
            ),
          ),
      ],
    );
  }

  // ── Result layout ─────────────────────────────────────────────────────────
  Widget _resultLayout(FullAnalysisResult r) => LayoutBuilder(
    builder: (ctx, box) {
      final wide = box.maxWidth > 740;
      final img = _annotatedImageWithNodes(r.imageBytes);
      final card = _diagnosticsPanel(
        r.diagnosticsTable,
        r.pixelSize,
        r.calibrationStatus,
        r.clinicalSummary,
      );
      if (wide) {
        return Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Expanded(flex: 3, child: img),
            const SizedBox(width: 16),
            SizedBox(width: 340, child: card),
          ],
        );
      }
      return Column(
        children: [
          SizedBox(height: box.maxHeight * 0.48, child: img),
          const SizedBox(height: 12),
          Expanded(child: card),
        ],
      );
    },
  );

  Widget _diagnosticsPanel(
    List<DiagnosticRow> rows,
    double pixelSize,
    String calibrationStatus,
    String clinicalSummary,
  ) => Container(
    decoration: BoxDecoration(
      color: _panel,
      borderRadius: BorderRadius.circular(16),
      border: Border.all(color: _border),
    ),
    child: Column(
      children: [
        _panelHeader(pixelSize, rows, calibrationStatus),
        const Divider(color: _border, height: 1),
        Expanded(
          child: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                if (clinicalSummary.isNotEmpty) _aiSummaryCard(clinicalSummary),
                if (_chatAnswer != null)
                  _chatResponseCard()
                else
                  ListView.separated(
                    padding: EdgeInsets.zero,
                    shrinkWrap: true,
                    physics: const NeverScrollableScrollPhysics(),
                    itemCount: rows.length,
                    separatorBuilder: (_, __) =>
                        const Divider(color: _border, height: 1),
                    itemBuilder: (_, i) => _diagRow(rows[i]),
                  ),
              ],
            ),
          ),
        ),
        if (clinicalSummary.isNotEmpty) _aiChatbotUI(),
      ],
    ),
  );

  Widget _aiSummaryCard(String summary) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: _blue.withOpacity(0.08),
        border: const Border(bottom: BorderSide(color: _border)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Text('🤖', style: TextStyle(fontSize: 14)),
              const SizedBox(width: 6),
              const Text(
                'AI Clinical Summary',
                style: TextStyle(
                  color: _blue,
                  fontWeight: FontWeight.w700,
                  fontSize: 13,
                ),
              ),
              const Spacer(),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                decoration: BoxDecoration(
                  color: _blue.withOpacity(0.15),
                  borderRadius: BorderRadius.circular(4),
                ),
                child: const Text(
                  'Llama 3',
                  style: TextStyle(
                    color: _blue,
                    fontSize: 9,
                    fontWeight: FontWeight.w800,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            summary,
            style: const TextStyle(
              color: Colors.white70,
              fontSize: 12,
              height: 1.4,
            ),
          ),
        ],
      ),
    );
  }

  Widget _chatResponseCard() {
    return Container(
      padding: const EdgeInsets.all(12),
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: _blue.withOpacity(0.1),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: _blue.withOpacity(0.3)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  '🤖 AI Response',
                  style: TextStyle(
                    color: _blue,
                    fontWeight: FontWeight.w700,
                    fontSize: 13,
                  ),
                ),
                InkWell(
                  onTap: () {
                    setState(() {
                      _chatAnswer = null;
                      _chatQuestion = ''; // Or empty? Yes, _chatQuestion = ''
                    });
                  },
                  borderRadius: BorderRadius.circular(12),
                  child: const Padding(
                    padding: EdgeInsets.all(2.0),
                    child: Icon(Icons.close, color: Colors.white54, size: 18),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),
            Text(
              _chatAnswer!,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 13,
                height: 1.4,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _aiChatbotUI() {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: const BoxDecoration(
        color: _card,
        border: Border(top: BorderSide(color: _border)),
        borderRadius: BorderRadius.only(
          bottomLeft: Radius.circular(16),
          bottomRight: Radius.circular(16),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          if (_isChatLoading)
            const Padding(
              padding: EdgeInsets.only(bottom: 10),
              child: Center(
                child: SizedBox(
                  width: 20,
                  height: 20,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    color: _blue,
                  ),
                ),
              ),
            ),
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _chatController,
                  style: const TextStyle(color: Colors.white, fontSize: 13),
                  decoration: InputDecoration(
                    hintText: 'Ask AI about this X-Ray...',
                    hintStyle: const TextStyle(
                      color: Colors.white38,
                      fontSize: 13,
                    ),
                    isDense: true,
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 10,
                    ),
                    filled: true,
                    fillColor: _dark,
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(8),
                      borderSide: BorderSide.none,
                    ),
                  ),
                  onSubmitted: (_) => _askAi(),
                ),
              ),
              const SizedBox(width: 8),
              InkWell(
                onTap: _isChatLoading ? null : _askAi,
                borderRadius: BorderRadius.circular(8),
                child: Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: _blue,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: const Icon(
                    Icons.send_rounded,
                    color: Colors.white,
                    size: 18,
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Future<void> _askAi() async {
    final q = _chatController.text.trim();
    if (q.isEmpty || _result == null) return;

    setState(() {
      _chatQuestion = q;
      _chatAnswer = null;
      _isChatLoading = true;
    });
    _chatController.clear();

    try {
      final answer = await ApiService.askQuestion(
        diagnostics: _result!.diagnosticsTable,
        question: q,
      );
      setState(() {
        _chatAnswer = answer;
        _isChatLoading = false;
      });
    } catch (e) {
      setState(() {
        _chatAnswer = 'Error: $e';
        _isChatLoading = false;
      });
    }
  }

  Widget _panelHeader(
    double pixelSize,
    List<DiagnosticRow> rows,
    String calibrationStatus,
  ) {
    final abnCount = rows.where((r) => r.isAbnormal).length;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: _teal.withOpacity(0.10),
        borderRadius: const BorderRadius.vertical(top: Radius.circular(16)),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ── Row 1: title + PDF icon (always fits) ───────────────────────
          Row(
            children: [
              const Icon(Icons.analytics_outlined, color: _teal, size: 18),
              const SizedBox(width: 8),
              const Expanded(
                child: Text(
                  'Clinical Report',
                  style: TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.w700,
                    fontSize: 14,
                  ),
                ),
              ),
              Tooltip(
                message: 'Export PDF Report',
                child: InkWell(
                  onTap: _exportPdf,
                  borderRadius: BorderRadius.circular(6),
                  child: Container(
                    padding: const EdgeInsets.all(5),
                    decoration: BoxDecoration(
                      color: _blue.withOpacity(0.15),
                      borderRadius: BorderRadius.circular(6),
                    ),
                    child: const Icon(
                      Icons.picture_as_pdf_rounded,
                      color: _blue,
                      size: 16,
                    ),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          // ── Row 2: chips in a Wrap (never overflows) ────────────────────
          Wrap(
            spacing: 6,
            runSpacing: 4,
            children: [
              if (abnCount > 0)
                _miniChip('$abnCount abnormal', _warn)
              else
                _miniChip('All normal', _normal),

              // Calibration validation feedback chip
              if (calibrationStatus == 'accepted')
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 3,
                  ),
                  decoration: BoxDecoration(
                    color: _green.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: _green.withOpacity(0.4)),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(Icons.check_circle, color: _green, size: 12),
                      const SizedBox(width: 4),
                      Text(
                        '${pixelSize.toStringAsFixed(3)} mm/px',
                        style: const TextStyle(
                          color: _green,
                          fontSize: 10,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ],
                  ),
                )
              else if (calibrationStatus == 'rejected')
                Tooltip(
                  message:
                      'Ruler dimension was implausible (>15% variance from dataset average). Used standard calibration fallback.',
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 8,
                      vertical: 3,
                    ),
                    decoration: BoxDecoration(
                      color: _warn.withOpacity(0.15),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: _warn.withOpacity(0.4)),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Icon(
                          Icons.warning_amber_rounded,
                          color: _warn,
                          size: 12,
                        ),
                        const SizedBox(width: 4),
                        Text(
                          '${pixelSize.toStringAsFixed(3)} mm/px',
                          style: const TextStyle(
                            color: _warn,
                            fontSize: 10,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ],
                    ),
                  ),
                )
              else if (calibrationStatus == 'accepted_no_csv')
                Tooltip(
                  message:
                      'Dataset default not available. Trusting manual ruler scale fully.',
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 8,
                      vertical: 3,
                    ),
                    decoration: BoxDecoration(
                      color: _green.withOpacity(0.15),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: _green.withOpacity(0.4)),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Icon(Icons.info_outline, color: _green, size: 12),
                        const SizedBox(width: 4),
                        Text(
                          '${pixelSize.toStringAsFixed(3)} mm/px',
                          style: const TextStyle(
                            color: _green,
                            fontSize: 10,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ],
                    ),
                  ),
                )
              else
                _miniChip(
                  '${pixelSize.toStringAsFixed(3)} mm/px',
                  Colors.white38,
                ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _diagRow(DiagnosticRow r) {
    final diffColor = r.isAbnormal ? _warn : _normal;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 3,
            height: 44,
            margin: const EdgeInsets.only(right: 10, top: 2),
            decoration: BoxDecoration(
              color: diffColor,
              borderRadius: BorderRadius.circular(2),
            ),
          ),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Text(
                        r.parameter,
                        style: const TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.w600,
                          fontSize: 13,
                        ),
                      ),
                    ),
                    Text(
                      r.value,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 13,
                        fontFeatures: [FontFeature.tabularFigures()],
                      ),
                    ),
                    const SizedBox(width: 10),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 6,
                        vertical: 2,
                      ),
                      decoration: BoxDecoration(
                        color: diffColor.withOpacity(0.15),
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: Text(
                        r.diff,
                        style: TextStyle(
                          color: diffColor,
                          fontSize: 11,
                          fontWeight: FontWeight.w700,
                          fontFeatures: const [FontFeature.tabularFigures()],
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 3),
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Norm: ${r.reference}  ',
                      style: const TextStyle(
                        color: Colors.white38,
                        fontSize: 11,
                      ),
                    ),
                    Expanded(
                      child: Text(
                        r.comment,
                        style: TextStyle(
                          color: r.isAbnormal
                              ? _warn.withOpacity(0.9)
                              : Colors.white54,
                          fontSize: 11,
                          fontStyle: r.isAbnormal
                              ? FontStyle.normal
                              : FontStyle.italic,
                        ),
                        softWrap: true,
                      ),
                    ),
                  ],
                ),
                // ── The Metric Gauge ──────────────────────────────────────────
                Builder(
                  builder: (ctx) {
                    final parsed = _parseDiagnosticData(r.value, r.reference);
                    if (parsed.actual != null &&
                        parsed.target != null &&
                        parsed.tolerance != null) {
                      return Padding(
                        padding: const EdgeInsets.only(top: 10.0, bottom: 2.0),
                        child: _DiagnosticGauge(
                          actual: parsed.actual!,
                          target: parsed.target!,
                          tolerance: parsed.tolerance!,
                          isAbnormal: r.isAbnormal,
                        ),
                      );
                    }
                    return const SizedBox.shrink();
                  },
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  ({double? actual, double? target, double? tolerance}) _parseDiagnosticData(
    String valueStr,
    String refStr,
  ) {
    final valMatch = RegExp(r'([+-]?\d+\.?\d*)').firstMatch(valueStr);
    double? actual = valMatch != null
        ? double.tryParse(valMatch.group(1)!)
        : null;

    final refMatches = RegExp(r'([+-]?\d+\.?\d*)').allMatches(refStr).toList();
    double? target;
    double? tolerance;

    if (refMatches.isNotEmpty) {
      target = double.tryParse(refMatches[0].group(1)!);
      if (refMatches.length > 1) {
        tolerance = double.tryParse(refMatches[1].group(1)!);
        if (tolerance != null) tolerance = tolerance.abs();
      }
    }

    // Fallback if tolerance is missing or zero
    if (tolerance == null || tolerance == 0) {
      tolerance = 2.0;
    }

    print(
      'Parsed Gauge [$valueStr -> $refStr] | actual: $actual, target: $target, tol: $tolerance',
    );

    return (actual: actual, target: target, tolerance: tolerance);
  }

  // ── Image panels ──────────────────────────────────────────────────────────
  Widget _emptyPanel() => GestureDetector(
    onTap: _pickImage,
    child: Container(
      decoration: BoxDecoration(
        color: _card,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: _blue.withOpacity(0.35), width: 1.5),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: _blue.withOpacity(0.12),
              shape: BoxShape.circle,
            ),
            child: const Icon(
              Icons.add_photo_alternate_outlined,
              color: _blue,
              size: 48,
            ),
          ),
          const SizedBox(height: 16),
          const Text(
            'Tap to Select X-Ray',
            style: TextStyle(
              color: Colors.white,
              fontSize: 16,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 6),
          const Text(
            'JPG · PNG · BMP',
            style: TextStyle(color: Colors.white38, fontSize: 12),
          ),
        ],
      ),
    ),
  );

  Widget _imageCard(Uint8List bytes, String badge, Color badgeColor) =>
      ClipRRect(
        borderRadius: BorderRadius.circular(16),
        child: Stack(
          fit: StackFit.expand,
          children: [
            Image.memory(
              bytes,
              fit: BoxFit.contain,
              width: double.infinity,
              height: double.infinity,
            ),
            Positioned(top: 10, right: 10, child: _chip(badge, badgeColor)),
          ],
        ),
      );

  // ── Annotated image + draggable landmark handles ──────────────────────────
  Widget _annotatedImageWithNodes(Uint8List bytes) {
    return LayoutBuilder(
      builder: (ctx, constraints) {
        // Backend image is at most 1024px wide; landmark coords are in that space.
        final rawW = (_nativeW ?? 1976).toDouble();
        final rawH = (_nativeH ?? 2225).toDouble();
        final nw = rawW > 1024 ? 1024.0 : rawW;
        final nh = rawW > 1024 ? (rawH * (1024.0 / rawW)) : rawH;

        final ww = constraints.maxWidth;
        final wh = constraints.maxHeight;

        // BoxFit.contain: uniform scale anchored to the limiting dimension.
        final scale = math.min(ww / nw, wh / nh);
        final offsetX = (ww - nw * scale) / 2.0;
        final offsetY = (wh - nh * scale) / 2.0;

        // Helper: derive LIVE scale+offset from the RenderBox at event time.
        // This avoids using stale LayoutBuilder captures after setState rebuilds.
        ({double sc, double ox, double oy, double nw, double nh}) _liveLayout() {
          final box = _annotatedStackKey.currentContext
              ?.findRenderObject() as RenderBox?;
          final liveW = box?.size.width ?? ww;
          final liveH = box?.size.height ?? wh;
          final liveSc = math.min(liveW / nw, liveH / nh);
          return (
            sc: liveSc,
            ox: (liveW - nw * liveSc) / 2.0,
            oy: (liveH - nh * liveSc) / 2.0,
            nw: nw,
            nh: nh,
          );
        }

        // ── Build landmark handles ──────────────────────────────────
        const nodeSize = 14.0;
        final handles = <Widget>[];
        if (_landmarks != null) {
          for (final entry in _landmarks!.entries) {
            final name = entry.key;
            final lm = entry.value;
            final sx = lm.x * scale + offsetX;
            final sy = lm.y * scale + offsetY;
            handles.add(
              Positioned(
                left: sx - nodeSize / 2,
                top: sy - nodeSize / 2,
                child: GestureDetector(
                  onPanStart: (d) {
                    final box = _annotatedStackKey.currentContext
                        ?.findRenderObject() as RenderBox?;
                    final local = box != null
                        ? box.globalToLocal(d.globalPosition)
                        : Offset(sx, sy);
                    setState(() {
                      _isDragging = true;
                      _activeLandmarkName = name;
                      _dragScreenPos = local;
                    });
                  },
                  onPanUpdate: (d) {
                    final box = _annotatedStackKey.currentContext
                        ?.findRenderObject() as RenderBox?;
                    if (box == null) return;
                    final local = box.globalToLocal(d.globalPosition);
                    // Re-derive layout from the LIVE RenderBox size, not the
                    // stale LayoutBuilder closure (avoids scatter after rebuild).
                    final live = _liveLayout();
                    final newX = ((local.dx - live.ox) / live.sc).clamp(0.0, live.nw);
                    final newY = ((local.dy - live.oy) / live.sc).clamp(0.0, live.nh);
                    setState(() {
                      _landmarks![name] = (x: newX, y: newY);
                      _dragScreenPos = local;
                    });
                  },
                  onPanEnd: (_) {
                    setState(() => _isDragging = false);
                    _adjustLandmarks();
                  },
                  child: Container(
                    width: nodeSize,
                    height: nodeSize,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: const Color(0x5500CFFF),
                      border: Border.all(
                        color: Colors.white.withOpacity(0.85),
                        width: 1.0,
                      ),
                    ),
                  ),
                ),
              ),
            );
          }
        }

        // ── Build Loupe overlay ────────────────────────────────────
        const loupeSize = 200.0;
        const zoom = 3.0;
        const loupeRadius = loupeSize / 2;
        const captureRadius = loupeSize / zoom; // screen-pixel threshold

        Widget? loupeWidget;
        if (_isDragging && _landmarks != null) {
          final lx = (_dragScreenPos.dx - loupeSize - 20).clamp(0.0, ww - loupeSize);
          final ly = (_dragScreenPos.dy - loupeSize - 20).clamp(0.0, wh - loupeSize);

          // ── Loupe X-ray image background ────────────────────────
          // The annotated image renders at nw*scale × nh*scale inside the stack,
          // offset by (offsetX, offsetY). In the loupe we render the image at
          // nw*scale*zoom × nh*scale*zoom so each image pixel is zoom times larger,
          // then translate so that the drag point maps to the loupe centre.
          final imgW = nw * scale * zoom;
          final imgH = nh * scale * zoom;
          // Where does _dragScreenPos sit within the unzoomed rendered image?
          final relX = _dragScreenPos.dx - offsetX; // 0..nw*scale
          final relY = _dragScreenPos.dy - offsetY; // 0..nh*scale
          final loupeImageLayer = Transform.translate(
            offset: Offset(
              loupeRadius - relX * zoom,  // place drag-x at loupe centre
              loupeRadius - relY * zoom,
            ),
            child: Image.memory(
              bytes,
              width: imgW,
              height: imgH,
              fit: BoxFit.fill, // fill the explicit pixel dimensions exactly
            ),
          );



          // ── Nearby landmark dots & labels ────────────────────────
          final loupeItems = <Widget>[];
          for (final entry in _landmarks!.entries) {
            final lmSx = entry.value.x * scale + offsetX;
            final lmSy = entry.value.y * scale + offsetY;
            final dx = lmSx - _dragScreenPos.dx;
            final dy = lmSy - _dragScreenPos.dy;
            final dist = math.sqrt(dx * dx + dy * dy);
            if (dist > captureRadius) continue;

            final lx2 = loupeRadius + dx * zoom;
            final ly2 = loupeRadius + dy * zoom;
            final isActive = entry.key == _activeLandmarkName;

            loupeItems.add(Positioned(
              left: lx2 - nodeSize / 2,
              top: ly2 - nodeSize / 2,
              child: Container(
                width: nodeSize,
                height: nodeSize,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: isActive
                      ? const Color(0xBB00CFFF)
                      : const Color(0x7700CFFF),
                  border: Border.all(
                    color: isActive ? Colors.white : Colors.white54,
                    width: isActive ? 1.5 : 1.0,
                  ),
                ),
              ),
            ));
            loupeItems.add(Positioned(
              left: lx2 + nodeSize / 2 + 2,
              top: ly2 - 7,
              child: Text(
                entry.key,
                style: TextStyle(
                  color: isActive ? Colors.white : Colors.white70,
                  fontSize: 11,
                  fontWeight: isActive ? FontWeight.bold : FontWeight.normal,
                  shadows: const [Shadow(color: Colors.black, blurRadius: 4)],
                ),
              ),
            ));
          }

          loupeWidget = Positioned(
            left: lx,
            top: ly,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Header
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: const Color(0xDD1A1A2E),
                    borderRadius: const BorderRadius.only(
                      topLeft: Radius.circular(10),
                      topRight: Radius.circular(10),
                    ),
                    border: Border.all(color: Colors.white24),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(Icons.zoom_in, color: Color(0xFF00CFFF), size: 13),
                      const SizedBox(width: 4),
                      Text(
                        '$_activeLandmarkName Precision View',
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 11,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ),
                ),
                // Loupe body: real X-ray + grid + dots + crosshair
                SizedBox(
                  width: loupeSize,
                  height: loupeSize,
                  child: DecoratedBox(
                    decoration: BoxDecoration(
                      border: Border.all(color: Colors.white24),
                      boxShadow: const [
                        BoxShadow(
                          color: Colors.black54,
                          blurRadius: 12,
                          offset: Offset(2, 4),
                        ),
                      ],
                    ),
                    child: ClipRect(
                      child: Stack(
                        children: [
                          // Layer 1: Zoomed X-ray image
                          loupeImageLayer,
                          // Layer 2: Precision grid
                          CustomPaint(
                            size: const Size(loupeSize, loupeSize),
                            painter: _LoupePainter(),
                          ),
                          // Layer 3: Nearby landmark dots + labels
                          ...loupeItems,
                          // Layer 4: Crosshair
                          Positioned(
                            left: loupeRadius - 0.5,
                            top: 0,
                            bottom: 0,
                            child: Container(width: 1, color: Colors.white30),
                          ),
                          Positioned(
                            top: loupeRadius - 0.5,
                            left: 0,
                            right: 0,
                            child: Container(height: 1, color: Colors.white30),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
                // Coordinate strip
                Container(
                  width: loupeSize,
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: const Color(0xEE0D0D1A),
                    borderRadius: const BorderRadius.only(
                      bottomLeft: Radius.circular(10),
                      bottomRight: Radius.circular(10),
                    ),
                    border: Border.all(color: Colors.white24),
                  ),
                  child: Builder(builder: (_) {
                    final lm = _landmarks![_activeLandmarkName];
                    if (lm == null) return const SizedBox.shrink();
                    return Text(
                      '$_activeLandmarkName: ${lm.x.toStringAsFixed(1)},  ${lm.y.toStringAsFixed(1)} px',
                      style: const TextStyle(
                        color: Color(0xFF00CFFF),
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                        fontFamily: 'monospace',
                      ),
                    );
                  }),
                ),
              ],
            ),
          );
        }

        return ClipRRect(
          borderRadius: BorderRadius.circular(16),
          child: Stack(
            key: _annotatedStackKey,
            fit: StackFit.expand,
            children: [
              Image.memory(
                bytes,
                fit: BoxFit.contain,
                width: double.infinity,
                height: double.infinity,
              ),
              if (_isAdjusting)
                Container(
                  color: Colors.black.withOpacity(0.25),
                  child: const Center(
                    child: SizedBox(
                      width: 48,
                      height: 48,
                      child: CircularProgressIndicator(
                        strokeWidth: 3,
                        valueColor: AlwaysStoppedAnimation<Color>(
                          Color(0xFF00CFFF),
                        ),
                      ),
                    ),
                  ),
                ),
              ...handles,
              if (loupeWidget != null) loupeWidget,
              Positioned(
                top: 10,
                right: 10,
                child: _chip('Analysis Complete', _teal),
              ),
            ],
          ),
        );
      },
    );
  }


  Widget _loadingPanel() => Container(
    decoration: BoxDecoration(
      color: _card,
      borderRadius: BorderRadius.circular(16),
    ),
    child: Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        const SizedBox(
          width: 72,
          height: 72,
          child: CircularProgressIndicator(
            strokeWidth: 4,
            valueColor: AlwaysStoppedAnimation<Color>(_teal),
          ),
        ),
        const SizedBox(height: 24),
        const Text(
          'AI is analyzing X-ray...',
          style: TextStyle(
            color: Colors.white,
            fontSize: 16,
            fontWeight: FontWeight.w600,
          ),
        ),
        const SizedBox(height: 8),
        const Padding(
          padding: EdgeInsets.symmetric(horizontal: 40),
          child: Text(
            'The first inference can take up to 60 seconds while the GPU warms up.',
            textAlign: TextAlign.center,
            style: TextStyle(color: Colors.white38, fontSize: 12),
          ),
        ),
      ],
    ),
  );

  // ── AppBar ─────────────────────────────────────────────────────────────────
  PreferredSizeWidget _appBar() => AppBar(
    backgroundColor: _dark,
    elevation: 0,
    title: Row(
      children: [
        Container(
          padding: const EdgeInsets.all(6),
          decoration: BoxDecoration(
            color: _blue.withOpacity(0.15),
            borderRadius: BorderRadius.circular(8),
          ),
          child: const Icon(Icons.biotech_rounded, color: _teal, size: 22),
        ),
        const SizedBox(width: 10),
        const Text(
          'Ceph AI Analysis',
          style: TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.w600,
            letterSpacing: 0.4,
          ),
        ),
      ],
    ),
    actions: [
      if (_result != null || _selectedBytes != null)
        IconButton(
          icon: const Icon(Icons.refresh_rounded, color: Colors.white54),
          tooltip: 'Reset',
          onPressed: _reset,
        ),
    ],
  );

  Widget _header() {
    final title = _result != null
        ? 'Analysis Complete'
        : _selectedBytes != null
        ? 'X-Ray Selected'
        : 'Upload X-Ray';
    final sub = _result != null
        ? 'Cephalometric tracing and clinical report shown below'
        : _selectedBytes != null
        ? 'Tap "Analyze" to scan image'
        : 'Select a lateral cephalogram from your gallery';

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 22,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 4),
        Text(sub, style: const TextStyle(color: Colors.white54, fontSize: 13)),
      ],
    );
  }

  // ── Error + Buttons ───────────────────────────────────────────────────────
  Widget _errorBanner() => Container(
    margin: const EdgeInsets.only(bottom: 12),
    padding: const EdgeInsets.all(14),
    decoration: BoxDecoration(
      color: const Color(0xFF4E1A1A),
      borderRadius: BorderRadius.circular(12),
      border: Border.all(color: Colors.redAccent.withOpacity(0.5)),
    ),
    child: Row(
      children: [
        const Icon(Icons.error_outline, color: Colors.redAccent, size: 20),
        const SizedBox(width: 10),
        Expanded(
          child: Text(
            _errorText!,
            style: const TextStyle(color: Colors.redAccent, fontSize: 13),
          ),
        ),
      ],
    ),
  );

  Widget _actionButtons() {
    if (_result != null) {
      return _primaryBtn(
        label: 'Analyze Another X-Ray',
        icon: Icons.add_photo_alternate_outlined,
        onTap: _pickImage,
        color: _blue,
      );
    }
    if (_selectedBytes != null) {
      return Column(
        children: [
          _primaryBtn(
            label: 'Analyze X-Ray',
            icon: Icons.biotech_rounded,
            onTap: _analyze,
            color: _teal,
          ),
          const SizedBox(height: 10),
          _secondaryBtn(label: 'Change Image', onTap: _pickImage),
        ],
      );
    }
    return _primaryBtn(
      label: 'Select X-Ray Image',
      icon: Icons.add_photo_alternate_outlined,
      onTap: _pickImage,
      color: _blue,
    );
  }

  // ── Reusable widgets ───────────────────────────────────────────────────────
  Widget _primaryBtn({
    required String label,
    required IconData icon,
    required VoidCallback onTap,
    required Color color,
  }) => ElevatedButton.icon(
    style: ElevatedButton.styleFrom(
      backgroundColor: color,
      foregroundColor: Colors.white,
      minimumSize: const Size.fromHeight(52),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      elevation: 0,
    ),
    icon: Icon(icon, size: 20),
    label: Text(
      label,
      style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600),
    ),
    onPressed: onTap,
  );

  Widget _secondaryBtn({required String label, required VoidCallback onTap}) =>
      OutlinedButton(
        style: OutlinedButton.styleFrom(
          foregroundColor: Colors.white70,
          minimumSize: const Size.fromHeight(48),
          side: const BorderSide(color: Colors.white24),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
        ),
        onPressed: onTap,
        child: Text(label),
      );

  Widget _chip(String label, Color color) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
    decoration: BoxDecoration(
      color: color.withOpacity(0.88),
      borderRadius: BorderRadius.circular(20),
    ),
    child: Text(
      label,
      style: const TextStyle(
        color: Colors.white,
        fontSize: 11,
        fontWeight: FontWeight.w600,
      ),
    ),
  );

  Widget _miniChip(String label, Color color) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
    decoration: BoxDecoration(
      color: color.withOpacity(0.15),
      borderRadius: BorderRadius.circular(12),
      border: Border.all(color: color.withOpacity(0.4)),
    ),
    child: Text(
      label,
      style: TextStyle(color: color, fontSize: 10, fontWeight: FontWeight.w700),
    ),
  );
}

// ── Custom Painter for Ruler ────────────────────────────────────────────────
class _RulerPainter extends CustomPainter {
  final Offset? p1;
  final Offset? p2;
  final Color color;
  _RulerPainter(this.p1, this.p2, this.color);

  @override
  void paint(Canvas canvas, Size size) {
    if (p1 == null) return;
    final paint = Paint()
      ..color = color
      ..strokeWidth = 2.5
      ..strokeCap = StrokeCap.round;

    final crossPaint = Paint()
      ..color = color
      ..strokeWidth = 2.0;

    void drawCross(Offset center) {
      canvas.drawLine(
        center - const Offset(6, 0),
        center + const Offset(6, 0),
        crossPaint,
      );
      canvas.drawLine(
        center - const Offset(0, 6),
        center + const Offset(0, 6),
        crossPaint,
      );
    }

    drawCross(p1!);
    if (p2 != null) {
      drawCross(p2!);
      canvas.drawLine(p1!, p2!, paint); // Use 'paint' for the line
      // Label
      final md = (p1! + p2!) / 2;
      final textSpan = TextSpan(
        text: 'Drag to set distance',
        style: TextStyle(
          color: color,
          fontSize: 12,
          fontWeight: FontWeight.bold,
          shadows: const [
            Shadow(color: Colors.black87, blurRadius: 4, offset: Offset(1, 1)),
          ],
        ),
      );
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();
      textPainter.paint(canvas, md - Offset(textPainter.width / 2, -10));
    }
  }

  @override
  bool shouldRepaint(covariant _RulerPainter old) =>
      p1 != old.p1 || p2 != old.p2 || color != old.color;
}

// ── Custom Painter for Diagnostic Gauge ─────────────────────────────────────
class _DiagnosticGauge extends StatelessWidget {
  final double actual;
  final double target;
  final double tolerance;
  final bool isAbnormal;

  const _DiagnosticGauge({
    required this.actual,
    required this.target,
    required this.tolerance,
    required this.isAbnormal,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      height: 12,
      child: CustomPaint(
        painter: _GaugePainter(
          actual: actual,
          target: target,
          tolerance: tolerance,
          isAbnormal: isAbnormal,
        ),
      ),
    );
  }
}

class _GaugePainter extends CustomPainter {
  final double actual;
  final double target;
  final double tolerance;
  final bool isAbnormal;

  _GaugePainter({
    required this.actual,
    required this.target,
    required this.tolerance,
    required this.isAbnormal,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final double rangeMultiplier = 3.0; // Show 3x tolerance on each side
    final double minVal = target - (tolerance * rangeMultiplier);
    final double maxVal = target + (tolerance * rangeMultiplier);
    final double range = maxVal - minVal;

    final Paint trackPaint = Paint()
      ..color = Colors.white.withOpacity(0.1)
      ..strokeWidth = 4
      ..strokeCap = StrokeCap.round;

    final Paint normalZonePaint = Paint()
      ..color = const Color(0xFF4CAF50)
          .withOpacity(0.4) // _normal green
      ..strokeWidth = 4
      ..strokeCap = StrokeCap.round;

    final Paint markerPaint = Paint()
      ..color = isAbnormal ? const Color(0xFFFF5252) : Colors.white
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round;

    final Paint centerLinePaint = Paint()
      ..color = Colors.white38
      ..strokeWidth = 1.5;

    final double w = size.width;
    final double h = size.height;
    final double cy = h / 2;

    if (w <= 4.0 || h <= 0.0) return; // Prevent sizing layout crash

    // 1. Draw track
    canvas.drawLine(Offset(0, cy), Offset(w, cy), trackPaint);

    // 2. Draw normal zone from (target - tolerance) to (target + tolerance)
    double normX1 = ((target - tolerance - minVal) / range) * w;
    double normX2 = ((target + tolerance - minVal) / range) * w;
    normX1 = normX1.clamp(0.0, w);
    normX2 = normX2.clamp(0.0, w);
    canvas.drawLine(Offset(normX1, cy), Offset(normX2, cy), normalZonePaint);

    // 3. Draw center tick at target
    double targetX = ((target - minVal) / range) * w;
    targetX = targetX.clamp(0.0, w);
    canvas.drawLine(
      Offset(targetX, cy - 3),
      Offset(targetX, cy + 3),
      centerLinePaint,
    );

    // 4. Draw actual patient marker
    double actualX = ((actual - minVal) / range) * w;
    actualX = actualX.clamp(2.0, w - 2.0); // Keep marker inside bounds
    canvas.drawLine(
      Offset(actualX, cy - 5),
      Offset(actualX, cy + 5),
      markerPaint,
    );
  }

  @override
  bool shouldRepaint(covariant _GaugePainter old) =>
      actual != old.actual ||
      target != old.target ||
      tolerance != old.tolerance ||
      isAbnormal != old.isAbnormal;
}

// ── Loupe grid painter ────────────────────────────────────────────────────────
class _LoupePainter extends CustomPainter {
  const _LoupePainter();

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = const Color(0x22FFFFFF)
      ..strokeWidth = 0.5;

    const step = 20.0;
    for (double x = 0; x <= size.width; x += step) {
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), paint);
    }
    for (double y = 0; y <= size.height; y += step) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), paint);
    }
  }

  @override
  bool shouldRepaint(covariant _LoupePainter old) => false;
}
