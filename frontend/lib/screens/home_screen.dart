import 'dart:math' as math;
import 'dart:async';
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

class _HomeScreenState extends State<HomeScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _glowController;
  late Animation<double> _glowAnimation;

  @override
  void initState() {
    super.initState();
    _glowController = AnimationController(
      duration: const Duration(milliseconds: 3000),
      vsync: this,
    )..repeat(reverse: true);
    _glowAnimation = Tween<double>(begin: 0.3, end: 0.6).animate(
      CurvedAnimation(parent: _glowController, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _glowController.dispose();
    _chatController.dispose();
    _loadingMessageTimer?.cancel();
    super.dispose();
  }

  // ── Loading messages for cycling ──────────────────────────────────────────
  static const _loadingMessages = [
    'Uploading Cephalogram...',
    'Detecting Sella and Nasion...',
    'Mapping Maxillary Landmarks...',
    'Calculating SNA & SNB Angles...',
    'Generating Clinical Report...',
  ];

  void _startLoadingMessageCycle() {
    _loadingMessageIndex = 0;
    _loadingMessageTimer?.cancel();
    _loadingMessageTimer = Timer.periodic(
      const Duration(milliseconds: 1500),
      (timer) {
        if (mounted && _isLoading) {
          setState(() {
            _loadingMessageIndex = (_loadingMessageIndex + 1) % _loadingMessages.length;
          });
        } else {
          timer.cancel();
        }
      },
    );
  }

  void _stopLoadingMessageCycle() {
    _loadingMessageTimer?.cancel();
    _loadingMessageTimer = null;
  }

  // ── Analysis state ─────────────────────────────────────────────────────────
  Uint8List? _selectedBytes;
  String? _selectedFilename;
  FullAnalysisResult? _result;
  bool _isLoading = false;
  bool _isAdjusting = false;
  String? _errorText;

  // Loading message cycling state
  int _loadingMessageIndex = 0;
  Timer? _loadingMessageTimer;

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
    
    // Immediately show loading state BEFORE heavy processing to prevent UI freeze
    setState(() {
      _selectedBytes = f.bytes;
      _selectedFilename = f.name;
      _result = _errorText = null;
      _nativeW = _nativeH = null;
      _chatQuestion = '';
      _chatAnswer = null;
      _isChatLoading = false;
      _chatController.clear();
      _isLoading = true;
    });
    
    // Small delay to ensure UI paints the loading state before heavy processing
    await Future.delayed(const Duration(milliseconds: 50));
    
    // Decode native image dimensions for accurate calibration mapping (heavy operation)
    if (f.bytes != null) {
      final codec = await ui.instantiateImageCodec(f.bytes!);
      final frame = await codec.getNextFrame();
      if (mounted) {
        setState(() {
          _nativeW = frame.image.width;
          _nativeH = frame.image.height;
          _isLoading = false; // Done with processing, hide loading
        });
      }
      frame.image.dispose();
    } else {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
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

  // ── Design tokens (Dark Slate with Accent Glows) ───────────────────────────
  static const _dark = Color(0xFF0A0E1A); // Deep slate/near-black background
  static const _card = Color(0xFF151A2E); // Slightly lighter card
  static const _panel = Color(0xFF0F1428); // Panel background
  static const _blue = Color(0xFF00B4D8); // Cyan accent
  static const _purple = Color(0xFF9D4EDD); // Purple accent glow
  static const _teal = Color(0xFF00FFA3); // Bright teal accent
  static const _border = Color(0xFF1E2A4A); // Subtle border
  static const _normal = Color(0xFF00FFA3); // Normal state
  static const _warn = Color(0xFFFF6B35); // Warning state
  static const _green = Color(0xFF76FF03); // Ruler line color
  static const _glassBg = Color(0x0DFFFFFF); // Glass background (5% white)
  static const _glassBorder = Color(0x26FFFFFF); // Glass border (15% white)

  // ── Analyze ───────────────────────────────────────────────────────────────
  Future<void> _analyze() async {
    if (_selectedBytes == null) return;
    
    // Step 1: Immediately set loading state
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

    // Step 2: CRITICAL - Force a frame render so loading UI appears
    // This prevents the 2-3 second freeze before loading screen shows
    await Future.delayed(const Duration(milliseconds: 100));
    
    if (!mounted) return;

    // Step 3: Start cycling messages AFTER the delay ensures UI is painted
    _startLoadingMessageCycle();

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
      _stopLoadingMessageCycle();
    } on ApiException catch (e) {
      _stopLoadingMessageCycle();
      if (mounted) setState(() => _errorText = e.message);
    } catch (e) {
      _stopLoadingMessageCycle();
      if (mounted) setState(() => _errorText = 'Unexpected error: $e');
    } finally {
      _stopLoadingMessageCycle();
      if (mounted) setState(() => _isLoading = false);
    }
  }

  // ── Adjust landmarks (re-render without PyTorch) ──────────────────────────
  Future<void> _adjustLandmarks() async {
    if (_selectedBytes == null || _landmarks == null || _result == null) return;
    
    setState(() => _isAdjusting = true);
    
    // Force a frame render so loading overlay appears
    await Future.delayed(const Duration(milliseconds: 50));
    
    if (!mounted) return;
    
    try {
      // Compress image in background before sending (prevents UI freeze)
      final compressedBytes = await ApiService.compressImageAsync(_selectedBytes!);
      
      final res = await ApiService.adjustLandmarks(
        imageBytes: compressedBytes,
        filename: _selectedFilename ?? 'image.jpg',
        landmarks: _landmarks!,
      );
      if (mounted) {
        setState(() {
          _result = res;
          _landmarks = Map.from(res.landmarks);
        });
      }
    } on ApiException catch (e) {
      if (mounted) setState(() => _errorText = e.message);
    } catch (e) {
      if (mounted) setState(() => _errorText = 'Unexpected error: $e');
    } finally {
      if (mounted) setState(() => _isAdjusting = false);
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
      child: Stack(
        children: [
          // Animated gradient glow blobs background
          _gradientGlowBackground(),
          // Main content
          Padding(
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
        ],
      ),
    ),
  );

  // ── Animated Gradient Glow Background ──────────────────────────────────────
  Widget _gradientGlowBackground() {
    return AnimatedBuilder(
      animation: _glowAnimation,
      builder: (context, child) {
        return Container(
          decoration: BoxDecoration(
            gradient: RadialGradient(
              colors: [
                _blue.withOpacity(_glowAnimation.value * 0.15),
                _purple.withOpacity(_glowAnimation.value * 0.1),
                Colors.transparent,
              ],
              stops: const [0.0, 0.5, 1.0],
              center: const Alignment(0.3, 0.3),
              radius: 1.5,
            ),
          ),
        );
      },
    );
  }

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
            child: ClipRRect(
              borderRadius: BorderRadius.circular(12),
              child: BackdropFilter(
                filter: ui.ImageFilter.blur(sigmaX: 15, sigmaY: 15),
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 14,
                    vertical: 10,
                  ),
                  decoration: BoxDecoration(
                    color: _dark.withOpacity(0.85),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: _green.withOpacity(0.5)),
                    boxShadow: [
                      BoxShadow(
                        color: _green.withOpacity(0.15),
                        blurRadius: 12,
                        spreadRadius: 0,
                      ),
                    ],
                  ),
                  child: Row(
                    children: [
                      Icon(Icons.ads_click, color: _green, size: 20),
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
                      ClipRRect(
                        borderRadius: BorderRadius.circular(8),
                        child: Material(
                          color: Colors.transparent,
                          child: InkWell(
                            onTap: _resetCalibration,
                            child: Container(
                              padding: const EdgeInsets.all(4),
                              child: const Icon(
                                Icons.close,
                                color: Colors.white54,
                                size: 20,
                              ),
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        // Confirmed Badge
        if (_calibration != null && !_isCalibrating)
          Positioned(
            top: 16,
            left: 16,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(12),
              child: BackdropFilter(
                filter: ui.ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 8,
                  ),
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
                      ClipRRect(
                        borderRadius: BorderRadius.circular(6),
                        child: Material(
                          color: Colors.transparent,
                          child: InkWell(
                            onTap: _resetCalibration,
                            child: Container(
                              padding: const EdgeInsets.all(4),
                              child: const Icon(
                                Icons.close,
                                color: _green,
                                size: 16,
                              ),
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        // Hovering "Calibrate Scale" Action Button Button
        if (!_isCalibrating && _calibration == null)
          Positioned(
            bottom: 20,
            left: 20,
            right: 20,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(12),
              child: Material(
                color: Colors.transparent,
                child: InkWell(
                  onTap: () => setState(() => _isCalibrating = true),
                  child: Container(
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    decoration: BoxDecoration(
                      color: _dark.withOpacity(0.85),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: _glassBorder),
                      boxShadow: [
                        BoxShadow(
                          color: _green.withOpacity(0.2),
                          blurRadius: 12,
                          spreadRadius: 0,
                        ),
                      ],
                    ),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Icon(Icons.straighten, size: 20, color: _green),
                        const SizedBox(width: 10),
                        const Text(
                          'Calibrate Scale (Optional)',
                          style: TextStyle(
                            fontSize: 15,
                            fontWeight: FontWeight.w600,
                            color: Colors.white,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
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
  ) => ClipRRect(
    borderRadius: BorderRadius.circular(16),
    child: BackdropFilter(
      filter: ui.ImageFilter.blur(sigmaX: 20, sigmaY: 20),
      child: Container(
        decoration: BoxDecoration(
          color: _glassBg,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: _glassBorder),
          boxShadow: [
            BoxShadow(
              color: _blue.withOpacity(0.1),
              blurRadius: 20,
              spreadRadius: 0,
            ),
          ],
        ),
        child: Column(
          children: [
            _panelHeader(pixelSize, rows, calibrationStatus),
            const Divider(color: _glassBorder, height: 1),
            Expanded(
              child: SingleChildScrollView(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    if (clinicalSummary.isNotEmpty)
                      _aiSummaryCard(clinicalSummary),
                    if (_chatAnswer != null)
                      _chatResponseCard()
                    else
                      ListView.separated(
                        padding: EdgeInsets.zero,
                        shrinkWrap: true,
                        physics: const NeverScrollableScrollPhysics(),
                        itemCount: rows.length,
                        separatorBuilder: (_, __) =>
                            const Divider(color: _glassBorder, height: 1),
                        itemBuilder: (_, i) => _diagRow(rows[i]),
                      ),
                  ],
                ),
              ),
            ),
            if (clinicalSummary.isNotEmpty) _aiChatbotUI(),
          ],
        ),
      ),
    ),
  );

  Widget _aiSummaryCard(String summary) {
    return ClipRRect(
      borderRadius: const BorderRadius.vertical(bottom: Radius.circular(16)),
      child: BackdropFilter(
        filter: ui.ImageFilter.blur(sigmaX: 15, sigmaY: 15),
        child: Container(
          width: double.infinity,
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: _blue.withOpacity(0.08),
            border: Border(bottom: BorderSide(color: _glassBorder)),
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
                  ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: BackdropFilter(
                      filter: ui.ImageFilter.blur(sigmaX: 8, sigmaY: 8),
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 6, vertical: 2),
                        decoration: BoxDecoration(
                          color: _blue.withOpacity(0.15),
                          borderRadius: BorderRadius.circular(4),
                          border: Border.all(color: _glassBorder),
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
        ),
      ),
    );
  }

  Widget _chatResponseCard() {
    return ClipRRect(
      borderRadius: BorderRadius.circular(12),
      child: BackdropFilter(
        filter: ui.ImageFilter.blur(sigmaX: 15, sigmaY: 15),
        child: Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: _blue.withOpacity(0.1),
            borderRadius: BorderRadius.circular(12),
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
                        _chatQuestion = '';
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
      ),
    );
  }

  Widget _aiChatbotUI() {
    return ClipRRect(
      borderRadius: const BorderRadius.vertical(bottom: Radius.circular(16)),
      child: BackdropFilter(
        filter: ui.ImageFilter.blur(sigmaX: 15, sigmaY: 15),
        child: Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: _glassBg,
            border: Border(top: BorderSide(color: _glassBorder)),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              if (_isChatLoading)
                Padding(
                  padding: const EdgeInsets.only(bottom: 10),
                  child: Center(
                    child: SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        valueColor: AlwaysStoppedAnimation<Color>(_blue),
                      ),
                    ),
                  ),
                ),
              Row(
                children: [
                  Expanded(
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: BackdropFilter(
                        filter: ui.ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 12,
                            vertical: 10,
                          ),
                          decoration: BoxDecoration(
                            color: _dark.withOpacity(0.5),
                            border:
                                Border.all(color: _glassBorder.withOpacity(0.5)),
                          ),
                          child: TextField(
                            controller: _chatController,
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 13,
                            ),
                            decoration: InputDecoration(
                              hintText: 'Ask AI about this X-Ray...',
                              hintStyle: const TextStyle(
                                color: Colors.white38,
                                fontSize: 13,
                              ),
                              isDense: true,
                              contentPadding: EdgeInsets.zero,
                              border: InputBorder.none,
                            ),
                            onSubmitted: (_) => _askAi(),
                          ),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(8),
                    child: Material(
                      color: Colors.transparent,
                      child: InkWell(
                        onTap: _isChatLoading ? null : _askAi,
                        child: Container(
                          padding: const EdgeInsets.all(8),
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              colors: [_blue, _teal],
                            ),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: const Icon(
                            Icons.send_rounded,
                            color: Colors.white,
                            size: 18,
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
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
    return ClipRRect(
      borderRadius: const BorderRadius.vertical(top: Radius.circular(16)),
      child: BackdropFilter(
        filter: ui.ImageFilter.blur(sigmaX: 15, sigmaY: 15),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
          decoration: BoxDecoration(
            color: _teal.withOpacity(0.10),
            border: Border(bottom: BorderSide(color: _glassBorder)),
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
                  ClipRRect(
                    borderRadius: BorderRadius.circular(6),
                    child: Material(
                      color: Colors.transparent,
                      child: InkWell(
                        onTap: _exportPdf,
                        child: Container(
                          padding: const EdgeInsets.all(5),
                          decoration: BoxDecoration(
                            color: _blue.withOpacity(0.15),
                            borderRadius: BorderRadius.circular(6),
                            border: Border.all(color: _glassBorder),
                          ),
                          child: const Icon(
                            Icons.picture_as_pdf_rounded,
                            color: _blue,
                            size: 16,
                          ),
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
                    ClipRRect(
                      borderRadius: BorderRadius.circular(12),
                      child: BackdropFilter(
                        filter: ui.ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 8,
                            vertical: 3,
                          ),
                          decoration: BoxDecoration(
                            color: _green.withOpacity(0.15),
                            borderRadius: BorderRadius.circular(12),
                            border:
                                Border.all(color: _green.withOpacity(0.4)),
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              const Icon(Icons.check_circle,
                                  color: _green, size: 12),
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
                      ),
                    )
                  else if (calibrationStatus == 'rejected')
                    Tooltip(
                      message:
                          'Ruler dimension was implausible (>15% variance from dataset average). Used standard calibration fallback.',
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(12),
                        child: BackdropFilter(
                          filter: ui.ImageFilter.blur(sigmaX: 10, sigmaY: 10),
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
                                const Icon(Icons.warning_amber_rounded,
                                    color: _warn, size: 12),
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
                        ),
                      ),
                    )
                  else if (calibrationStatus == 'accepted_no_csv')
                    Tooltip(
                      message:
                          'Dataset default not available. Trusting manual ruler scale fully.',
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(12),
                        child: BackdropFilter(
                          filter: ui.ImageFilter.blur(sigmaX: 10, sigmaY: 10),
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
                                const Icon(Icons.info_outline,
                                    color: _green, size: 12),
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
        ),
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
              boxShadow: [
                BoxShadow(
                  color: diffColor.withOpacity(0.3),
                  blurRadius: 8,
                  spreadRadius: 0,
                ),
              ],
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
                    ClipRRect(
                      borderRadius: BorderRadius.circular(6),
                      child: BackdropFilter(
                        filter: ui.ImageFilter.blur(sigmaX: 8, sigmaY: 8),
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 6,
                            vertical: 2,
                          ),
                          decoration: BoxDecoration(
                            color: diffColor.withOpacity(0.15),
                            borderRadius: BorderRadius.circular(6),
                            border: Border.all(
                              color: diffColor.withOpacity(0.3),
                            ),
                          ),
                          child: Text(
                            r.diff,
                            style: TextStyle(
                              color: diffColor,
                              fontSize: 11,
                              fontWeight: FontWeight.w700,
                              fontFeatures:
                                  const [FontFeature.tabularFigures()],
                            ),
                          ),
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
    child: ClipRRect(
      borderRadius: BorderRadius.circular(16),
      child: BackdropFilter(
        filter: ui.ImageFilter.blur(sigmaX: 20, sigmaY: 20),
        child: Container(
          decoration: BoxDecoration(
            color: _glassBg,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: _blue.withOpacity(0.35), width: 1.5),
            boxShadow: [
              BoxShadow(
                color: _blue.withOpacity(0.15),
                blurRadius: 20,
                spreadRadius: 0,
              ),
            ],
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              ClipOval(
                child: BackdropFilter(
                  filter: ui.ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                  child: Container(
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: _blue.withOpacity(0.12),
                      shape: BoxShape.circle,
                      border: Border.all(color: _glassBorder),
                    ),
                    child: const Icon(
                      Icons.add_photo_alternate_outlined,
                      color: _blue,
                      size: 48,
                    ),
                  ),
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
  // ── Landmark Color Categories ──────────────────────────────────────────────
  Color _categoryFill(String name) {
    if (name.contains('Incisor') || name.contains('Molar') || name.contains('PM') || name.contains('Occlusal')) {
      return const Color(0xFFFFD740); // Dental
    }
    if (name.contains('Soft Tissue') || name.contains('Pronasale') || name.contains('Subnasale') || name.contains('Labrale')) {
      return const Color(0xFFEF5350); // Soft Tissue
    }
    return const Color(0xFF29B6F6); // Skeletal (Default)
  }

  Color _categoryStroke(String name) {
    if (name.contains('Incisor') || name.contains('Molar') || name.contains('PM') || name.contains('Occlusal')) {
      return const Color(0xFFFFE57F); // Dental
    }
    if (name.contains('Soft Tissue') || name.contains('Pronasale') || name.contains('Subnasale') || name.contains('Labrale')) {
      return const Color(0xFFFF8A80); // Soft Tissue
    }
    return const Color(0xFF81D4FA); // Skeletal (Default)
  }

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
                  child: Stack(
                    alignment: Alignment.center,
                    children: [
                      // Solid landmark dot with category colors
                      Container(
                        width: 10,
                        height: 10,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: _categoryFill(name),
                          border: Border.all(
                            color: _categoryStroke(name),
                            width: 1.5,
                          ),
                        ),
                      ),
                      // Invisible larger hit target
                      Container(
                        width: nodeSize,
                        height: nodeSize,
                        color: Colors.transparent,
                      ),
                    ],
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
          // The annotated image renders at nw*scale × nh*scale, offset by
          // (offsetX, offsetY). In the loupe we render the image at
          // nw*scale*zoom × nh*scale*zoom, then position it so the drag-point
          // maps to the loupe centre. Using Positioned (not Transform) so the
          // Stack gives the image its full zoomed dimensions.
          final imgW = nw * scale * zoom;
          final imgH = nh * scale * zoom;
          final relX = _dragScreenPos.dx - offsetX;
          final relY = _dragScreenPos.dy - offsetY;

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
              child: Stack(
                alignment: Alignment.center,
                children: [
                  // Solid landmark dot in loupe
                  Container(
                    width: 10,
                    height: 10,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: _categoryFill(entry.key),
                      border: Border.all(
                        color: _categoryStroke(entry.key),
                        width: 1.5,
                      ),
                    ),
                  ),
                  // Highlight ring if active
                  if (isActive)
                    Container(
                      width: nodeSize + 4,
                      height: nodeSize + 4,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        border: Border.all(
                          color: Colors.white,
                          width: 1.5,
                        ),
                      ),
                    ),
                ],
              ),
            ));
            loupeItems.add(Positioned(
              left: lx2 + nodeSize / 2 + 2,
              top: ly2 - 7,
              child: ClipRRect(
                borderRadius: BorderRadius.circular(4),
                child: BackdropFilter(
                  filter: ui.ImageFilter.blur(sigmaX: 5, sigmaY: 5),
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 4,
                      vertical: 2,
                    ),
                    decoration: BoxDecoration(
                      color: _dark.withOpacity(0.6),
                      borderRadius: BorderRadius.circular(4),
                      border: Border.all(color: _glassBorder),
                    ),
                    child: Text(
                      entry.key,
                      style: TextStyle(
                        color: isActive ? Colors.white : Colors.white70,
                        fontSize: 11,
                        fontWeight: isActive ? FontWeight.bold : FontWeight.normal,
                      ),
                    ),
                  ),
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
                ClipRRect(
                  borderRadius: const BorderRadius.vertical(
                    top: Radius.circular(10),
                  ),
                  child: BackdropFilter(
                    filter: ui.ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 8,
                        vertical: 3,
                      ),
                      decoration: BoxDecoration(
                        color: _dark.withOpacity(0.8),
                        border: Border.all(color: _glassBorder),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            Icons.zoom_in,
                            color: _blue,
                            size: 13,
                          ),
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
                  ),
                ),
                // Loupe body: real X-ray + grid + dots + crosshair
                ClipRRect(
                  borderRadius: BorderRadius.circular(10),
                  child: BackdropFilter(
                    filter: ui.ImageFilter.blur(sigmaX: 5, sigmaY: 5),
                    child: Container(
                      width: loupeSize,
                      height: loupeSize,
                      decoration: BoxDecoration(
                        border: Border.all(color: _glassBorder),
                        boxShadow: [
                          BoxShadow(
                            color: _blue.withOpacity(0.2),
                            blurRadius: 12,
                            spreadRadius: 0,
                          ),
                        ],
                      ),
                      child: ClipRect(
                        child: Stack(
                          clipBehavior: Clip.hardEdge,
                          children: [
                            // Layer 1: Zoomed X-ray image (Positioned gives full zoomed dimensions)
                            Positioned(
                              left: loupeRadius - relX * zoom,
                              top: loupeRadius - relY * zoom,
                              width: imgW,
                              height: imgH,
                              child: Image.memory(bytes, fit: BoxFit.fill),
                            ),
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
                ),
                // Coordinate strip
                ClipRRect(
                  borderRadius: const BorderRadius.vertical(
                    bottom: Radius.circular(10),
                  ),
                  child: BackdropFilter(
                    filter: ui.ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                    child: Container(
                      width: loupeSize,
                      padding: const EdgeInsets.symmetric(
                        horizontal: 8,
                        vertical: 4,
                      ),
                      decoration: BoxDecoration(
                        color: _dark.withOpacity(0.85),
                        border: Border.all(color: _glassBorder),
                      ),
                      child: Builder(builder: (_) {
                        final lm = _landmarks![_activeLandmarkName];
                        if (lm == null) return const SizedBox.shrink();
                        return Text(
                          '$_activeLandmarkName: ${lm.x.toStringAsFixed(1)},  ${lm.y.toStringAsFixed(1)} px',
                          style: TextStyle(
                            color: _blue,
                            fontSize: 11,
                            fontWeight: FontWeight.w600,
                            fontFamily: 'monospace',
                          ),
                        );
                      }),
                    ),
                  ),
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


  Widget _loadingPanel() => ClipRRect(
    borderRadius: BorderRadius.circular(16),
    child: BackdropFilter(
      filter: ui.ImageFilter.blur(sigmaX: 20, sigmaY: 20),
      child: Container(
        decoration: BoxDecoration(
          color: _glassBg,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: _glassBorder),
          boxShadow: [
            BoxShadow(
              color: _teal.withOpacity(0.15),
              blurRadius: 20,
              spreadRadius: 0,
            ),
          ],
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Animated scanning circle with pulsing glow
            SizedBox(
              width: 100,
              height: 100,
              child: Stack(
                alignment: Alignment.center,
                children: [
                  // Pulsing outer glow
                  AnimatedBuilder(
                    animation: _glowAnimation,
                    builder: (context, child) {
                      return Container(
                        width: 80 + (_glowAnimation.value * 20),
                        height: 80 + (_glowAnimation.value * 20),
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          gradient: RadialGradient(
                            colors: [
                              _teal.withOpacity(0.3 * _glowAnimation.value),
                              _teal.withOpacity(0.1 * _glowAnimation.value),
                              Colors.transparent,
                            ],
                          ),
                        ),
                      );
                    },
                  ),
                  // Inner scanning ring
                  AnimatedBuilder(
                    animation: _glowAnimation,
                    builder: (context, child) {
                      return Transform.rotate(
                        angle: _glowAnimation.value * 2 * 3.14159,
                        child: Container(
                          width: 60,
                          height: 60,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            border: Border.all(
                              color: _teal.withOpacity(0.8),
                              width: 3,
                            ),
                          ),
                          child: CustomPaint(
                            painter: _ScannerArcPainter(
                              progress: _glowAnimation.value,
                              color: _teal,
                            ),
                          ),
                        ),
                      );
                    },
                  ),
                  // Center X-ray icon placeholder
                  Container(
                    width: 36,
                    height: 36,
                    decoration: BoxDecoration(
                      color: _teal.withOpacity(0.2),
                      shape: BoxShape.circle,
                    ),
                    child: const Icon(
                      Icons.image_outlined,
                      color: _teal,
                      size: 20,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 28),
            // Cycling status message
            AnimatedSwitcher(
              duration: const Duration(milliseconds: 400),
              child: Text(
                _loadingMessages[_loadingMessageIndex],
                key: ValueKey<int>(_loadingMessageIndex),
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
            const SizedBox(height: 12),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 40),
              child: Text(
                'The first inference can take up to 60 seconds while the GPU warms up.',
                textAlign: TextAlign.center,
                style: TextStyle(color: Colors.white.withOpacity(0.38), fontSize: 12),
              ),
            ),
          ],
        ),
      ),
    ),
  );

  // ── AppBar ─────────────────────────────────────────────────────────────────
  PreferredSizeWidget _appBar() => AppBar(
    backgroundColor: Colors.transparent,
    elevation: 0,
    surfaceTintColor: Colors.transparent,
    title: Row(
      children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(10),
          child: BackdropFilter(
            filter: ui.ImageFilter.blur(sigmaX: 10, sigmaY: 10),
            child: Container(
              padding: const EdgeInsets.all(6),
              decoration: BoxDecoration(
                color: _glassBg,
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: _glassBorder),
              ),
              child: const Icon(Icons.biotech_rounded, color: _teal, size: 22),
            ),
          ),
        ),
        const SizedBox(width: 10),
        ClipRRect(
          borderRadius: BorderRadius.circular(10),
          child: BackdropFilter(
            filter: ui.ImageFilter.blur(sigmaX: 15, sigmaY: 15),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
              decoration: BoxDecoration(
                color: _glassBg,
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: _glassBorder),
              ),
              child: const Text(
                'Ceph AI Analysis',
                style: TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 0.4,
                ),
              ),
            ),
          ),
        ),
      ],
    ),
    actions: [
      if (_result != null || _selectedBytes != null)
        ClipRRect(
          borderRadius: BorderRadius.circular(8),
          child: Material(
            color: Colors.transparent,
            child: InkWell(
              onTap: _reset,
              borderRadius: BorderRadius.circular(8),
              child: Container(
                padding: const EdgeInsets.all(8),
                child: const Icon(
                  Icons.refresh_rounded,
                  color: Colors.white54,
                ),
              ),
            ),
          ),
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

    return ClipRRect(
      borderRadius: BorderRadius.circular(12),
      child: BackdropFilter(
        filter: ui.ImageFilter.blur(sigmaX: 10, sigmaY: 10),
        child: Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: _glassBg,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: _glassBorder),
          ),
          child: Column(
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
              Text(
                sub,
                style: const TextStyle(color: Colors.white54, fontSize: 13),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ── Error + Buttons ───────────────────────────────────────────────────────
  Widget _errorBanner() => ClipRRect(
    borderRadius: BorderRadius.circular(12),
    child: BackdropFilter(
      filter: ui.ImageFilter.blur(sigmaX: 15, sigmaY: 15),
      child: Container(
        margin: const EdgeInsets.only(bottom: 12),
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: const Color(0xFF4E1A1A).withOpacity(0.9),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.redAccent.withOpacity(0.5)),
          boxShadow: [
            BoxShadow(
              color: Colors.redAccent.withOpacity(0.2),
              blurRadius: 12,
              spreadRadius: 0,
            ),
          ],
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
      ),
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
  }) => ClipRRect(
    borderRadius: BorderRadius.circular(14),
    child: Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 16),
          decoration: BoxDecoration(
            gradient: LinearGradient(colors: [color, color.withOpacity(0.8)]),
            borderRadius: BorderRadius.circular(14),
            boxShadow: [
              BoxShadow(
                color: color.withOpacity(0.4),
                blurRadius: 12,
                spreadRadius: 0,
              ),
            ],
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon, size: 20, color: Colors.white),
              const SizedBox(width: 10),
              Text(
                label,
                style: const TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w600,
                  color: Colors.white,
                ),
              ),
            ],
          ),
        ),
      ),
    ),
  );

  Widget _secondaryBtn({required String label, required VoidCallback onTap}) =>
      ClipRRect(
        borderRadius: BorderRadius.circular(14),
        child: Material(
          color: Colors.transparent,
          child: InkWell(
            onTap: onTap,
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 14),
              decoration: BoxDecoration(
                color: _glassBg,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: _glassBorder),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.folder_open_outlined,
                      size: 18, color: Colors.white70),
                  const SizedBox(width: 10),
                  Text(
                    label,
                    style: const TextStyle(
                      fontSize: 15,
                      fontWeight: FontWeight.w600,
                      color: Colors.white70,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      );

  Widget _chip(String label, Color color) => ClipRRect(
    borderRadius: BorderRadius.circular(20),
    child: BackdropFilter(
      filter: ui.ImageFilter.blur(sigmaX: 8, sigmaY: 8),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: color.withOpacity(0.88),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: _glassBorder),
          boxShadow: [
            BoxShadow(
              color: color.withOpacity(0.3),
              blurRadius: 8,
              spreadRadius: 0,
            ),
          ],
        ),
        child: Text(
          label,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 11,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
    ),
  );

  Widget _miniChip(String label, Color color) => ClipRRect(
    borderRadius: BorderRadius.circular(12),
    child: BackdropFilter(
      filter: ui.ImageFilter.blur(sigmaX: 8, sigmaY: 8),
      child: Container(
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
      ),
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

// Custom painter for the scanning arc animation
class _ScannerArcPainter extends CustomPainter {
  final double progress;
  final Color color;

  _ScannerArcPainter({required this.progress, required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = 4
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    final center = Offset(size.width / 2, size.height / 2);
    final radius = size.width / 2 - 4;

    // Draw arc from 0 to progress * 360 degrees
    final sweepAngle = progress * 2 * 3.14159;
    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      -3.14159 / 2, // Start from top
      sweepAngle,
      false,
      paint,
    );
  }

  @override
  bool shouldRepaint(covariant _ScannerArcPainter old) =>
      old.progress != progress || old.color != color;
}
