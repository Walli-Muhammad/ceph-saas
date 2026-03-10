import 'package:flutter/material.dart';
import 'screens/home_screen.dart';

void main() {
  runApp(const CephAiApp());
}

class CephAiApp extends StatelessWidget {
  const CephAiApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Ceph AI Analysis',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: const Color(0xFF1A73E8),
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF1C1C2E),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFF1C1C2E),
          elevation: 0,
          surfaceTintColor: Colors.transparent,
        ),
        fontFamily: 'Roboto',
      ),
      home: const HomeScreen(),
    );
  }
}
