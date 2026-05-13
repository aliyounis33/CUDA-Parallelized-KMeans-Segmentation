#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QFrame>
#include <QSplitter>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QAbstractItemView>
#include <QHeaderView>
#include <QPainter>
#include <QLinearGradient>
#include <QFont>
#include <QPushButton>
#include <QLabel>
#include <QFileDialog>
#include <QPixmap>
#include <QImage>
#include <QElapsedTimer>
#include <QSpinBox>
#include <QComboBox>
#include <QMessageBox>
#include <QGraphicsOpacityEffect>
#include <QPropertyAnimation>
#include <QEasingCurve>
#include <QTimer>
#include "kernel.h"

// =====================================================================
// MAIN WINDOW CLASS DEFINITION
// =====================================================================
class MainWindow : public QMainWindow {
public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow() = default;

private:
    // --- UI Components ---
    QPushButton *btnLoad, *btnRunCPU, *btnRunGPU;
    QComboBox *comboMethod;
    QSpinBox *spinK;
    QLabel *lblImgOrig, *lblImgCPU, *lblImgGPU;
    QLabel *lblTimeCPU, *lblTimeGPU;
    QLabel *lblStatus;
    QLabel *lblChart;
    QLabel *lblWorkflow;
    QTableWidget *tblStats;

    QVector<QPropertyAnimation*> introAnimations;

    // --- State Variables ---
    QImage originalImage;
    int lastCpuMs = -1;
    int lastGpuMs = -1;
    QSize lastImageSize;

    // --- Core Methods ---
    void setupUI();
    void connectSignals();
    void loadImage();
    void runSegmentation(bool useGPU);
    
    // --- Helper Methods ---
    void createImageCard(QVBoxLayout* layout, QLabel*& imgLbl, QLabel*& timeLbl, const QString& title, const QString& subtitle);
    QPixmap createPerformanceChart(int cpuMs, int gpuMs);
    QPixmap createWorkflowFigure();
    void updatePerformanceUI(bool useGPU, int elapsedMs);
    void resetPerformanceUI();
    void applyIntroAnimation(QWidget* widget, int delayMs);
};

// =====================================================================
// MAIN WINDOW IMPLEMENTATION
// =====================================================================

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    setWindowTitle("CUDA K-Means Segmentation Profiler");
    resize(1200, 720);
    
    setupUI();
    connectSignals();
}

void MainWindow::setupUI() {
    QWidget *centralWidget = new QWidget(this);
    centralWidget->setObjectName("root");
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);
    mainLayout->setSpacing(18);
    mainLayout->setContentsMargins(20, 20, 20, 20);

    centralWidget->setStyleSheet(
        "QWidget#root { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #fef7ef, stop:0.45 #f2f7ff, stop:1 #eefbf6);"
        " color: #172131; font-family: 'Space Grotesk', 'Barlow', 'Tahoma'; }"
        "QGroupBox { background: #ffffff; border: 1px solid #e2e7f1; border-radius: 14px; margin-top: 14px; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 16px; padding: 2px 8px; color: #1b2536; font-weight: 700; letter-spacing: 0.2px; }"
        "QFrame#card { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #f6f8fb);"
        " border: 1px solid #e2e7f1; border-radius: 16px; }"
        "QPushButton { border: none; border-radius: 10px; padding: 9px 16px; font-weight: 700; letter-spacing: 0.3px; }"
        "QPushButton#primary { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0ea5a8, stop:1 #14b8a6); color: #ffffff; }"
        "QPushButton#primary:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #14b8a6, stop:1 #2dd4bf); }"
        "QPushButton#primary:pressed { background: #0f766e; }"
        "QPushButton#primaryAlt { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f59e0b, stop:1 #f97316); color: #ffffff; }"
        "QPushButton#primaryAlt:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #fbbf24, stop:1 #fb923c); }"
        "QPushButton#primaryAlt:pressed { background: #c2410c; }"
        "QPushButton#secondary { background: #f1f5ff; color: #1d4ed8; border: 1px solid #c7d2fe; }"
        "QPushButton#secondary:hover { background: #e0e7ff; }"
        "QPushButton:disabled { background: #c6cfde; color: #f8fafc; }"
        "QComboBox, QSpinBox { background: #ffffff; border: 1px solid #cad4e5; border-radius: 8px; padding: 7px 10px; color: #1f2937; }"
        "QComboBox QAbstractItemView { background: #ffffff; color: #1f2937; selection-background-color: #e0f2fe; selection-color: #0f172a; }"
        "QLabel { color: #1f2937; }"
        "QLabel#title { font-size: 24px; font-weight: 800; letter-spacing: 0.4px; }"
        "QLabel#subtitle { color: #5b6475; font-size: 13px; }"
        "QLabel#badge { background: #e6fffb; color: #0f766e; border-radius: 999px; padding: 5px 12px; font-weight: 700; }"
        "QTableWidget { background: #ffffff; border: 1px solid #e2e7f1; border-radius: 10px; gridline-color: #eef1f6; }"
        "QTableWidget::item { padding: 6px; color: #1f2937; }"
        "QHeaderView::section { background: #f2f5fb; padding: 7px; border: none; font-weight: 700; color: #1f2a3a; }"
    );

    // --- 1. Header ---
    QFrame *header = new QFrame();
    header->setObjectName("card");
    QHBoxLayout *headerLayout = new QHBoxLayout(header);
    headerLayout->setContentsMargins(16, 12, 16, 12);

    QVBoxLayout *headerTextLayout = new QVBoxLayout();
    QLabel *title = new QLabel("CUDA K-Means Segmentation Profiler");
    title->setObjectName("title");
    QLabel *subtitle = new QLabel("Compare CPU vs GPU segmentation performance.");
    subtitle->setObjectName("subtitle");
    headerTextLayout->addWidget(title);
    headerTextLayout->addWidget(subtitle);

    lblStatus = new QLabel("Ready");
    lblStatus->setObjectName("badge");
    lblStatus->setAlignment(Qt::AlignCenter);

    headerLayout->addLayout(headerTextLayout);
    headerLayout->addStretch();
    headerLayout->addWidget(lblStatus);
    mainLayout->addWidget(header);

    // --- 2. Main Content Split ---
    QSplitter *splitter = new QSplitter(Qt::Horizontal);
    splitter->setChildrenCollapsible(false);

    // Left panel: controls + figures + stats
    QWidget *leftPanel = new QWidget();
    QVBoxLayout *leftLayout = new QVBoxLayout(leftPanel);
    leftLayout->setSpacing(14);
    leftLayout->setContentsMargins(0, 0, 0, 0);

    QGroupBox *controlGroup = new QGroupBox("Controls");
    controlGroup->setObjectName("panel");
    QGridLayout *controlLayout = new QGridLayout(controlGroup);
    controlLayout->setHorizontalSpacing(10);
    controlLayout->setVerticalSpacing(10);

    btnLoad = new QPushButton("Load Image");
    btnLoad->setObjectName("secondary");

    comboMethod = new QComboBox();
    comboMethod->addItems({"Basic K-Means", "Tiled K-Means", "Fuzzy C-Means", "Mini-Batch K-Means"});

    spinK = new QSpinBox();
    spinK->setRange(2, 32);
    spinK->setValue(3);

    btnRunCPU = new QPushButton("Run on CPU");
    btnRunGPU = new QPushButton("Run on GPU");
    btnRunCPU->setObjectName("primary");
    btnRunGPU->setObjectName("primaryAlt");
    btnRunCPU->setEnabled(false);
    btnRunGPU->setEnabled(false);

    controlLayout->addWidget(btnLoad, 0, 0, 1, 2);
    controlLayout->addWidget(new QLabel("Method"), 1, 0);
    controlLayout->addWidget(comboMethod, 1, 1);
    controlLayout->addWidget(new QLabel("Clusters (K)"), 2, 0);
    controlLayout->addWidget(spinK, 2, 1);
    controlLayout->addWidget(btnRunCPU, 3, 0);
    controlLayout->addWidget(btnRunGPU, 3, 1);

    QGroupBox *perfGroup = new QGroupBox("Performance Dashboard");
    perfGroup->setObjectName("panel");
    QVBoxLayout *perfLayout = new QVBoxLayout(perfGroup);
    tblStats = new QTableWidget(4, 2);
    tblStats->setObjectName("stats");
    tblStats->setHorizontalHeaderLabels({"Metric", "Value"});
    tblStats->verticalHeader()->setVisible(false);
    tblStats->setEditTriggers(QAbstractItemView::NoEditTriggers);
    tblStats->setSelectionMode(QAbstractItemView::NoSelection);
    tblStats->horizontalHeader()->setStretchLastSection(true);
    tblStats->setRowHeight(0, 26);
    tblStats->setRowHeight(1, 26);
    tblStats->setRowHeight(2, 26);
    tblStats->setRowHeight(3, 26);

    tblStats->setItem(0, 0, new QTableWidgetItem("GPU time"));
    tblStats->setItem(1, 0, new QTableWidgetItem("Speedup"));
    tblStats->setItem(2, 0, new QTableWidgetItem("Image size"));
    tblStats->setItem(3, 0, new QTableWidgetItem("Method"));

    for (int row = 0; row < 4; ++row) {
        tblStats->setItem(row, 1, new QTableWidgetItem("--"));
    }

    lblChart = new QLabel();
    lblChart->setAlignment(Qt::AlignCenter);
    lblChart->setPixmap(createPerformanceChart(-1, -1));

    perfLayout->addWidget(tblStats);
    perfLayout->addWidget(lblChart);

    leftLayout->addWidget(controlGroup);
    leftLayout->addWidget(perfGroup);
    leftLayout->addStretch();

    // Right panel: image results
    QWidget *rightPanel = new QWidget();
    QVBoxLayout *rightLayout = new QVBoxLayout(rightPanel);
    rightLayout->setSpacing(14);
    rightLayout->setContentsMargins(0, 0, 0, 0);

    QFrame *imageCard = new QFrame();
    imageCard->setObjectName("card");
    QGridLayout *imageGrid = new QGridLayout(imageCard);
    imageGrid->setContentsMargins(16, 16, 16, 16);
    imageGrid->setHorizontalSpacing(12);
    imageGrid->setVerticalSpacing(12);

    QVBoxLayout *layoutOrig = new QVBoxLayout();
    QVBoxLayout *layoutCPU = new QVBoxLayout();
    QVBoxLayout *layoutGPU = new QVBoxLayout();
    QLabel *lblTimeOrig;

    createImageCard(layoutOrig, lblImgOrig, lblTimeOrig, "Original", "Input scan");
    createImageCard(layoutCPU, lblImgCPU, lblTimeCPU, "CPU Result", "Baseline segmentation");
    createImageCard(layoutGPU, lblImgGPU, lblTimeGPU, "GPU Result", "CUDA acceleration");
    lblTimeOrig->hide();

    imageGrid->addLayout(layoutOrig, 0, 0, 1, 2);
    imageGrid->addLayout(layoutCPU, 1, 0);
    imageGrid->addLayout(layoutGPU, 1, 1);

    rightLayout->addWidget(imageCard);

    splitter->addWidget(leftPanel);
    splitter->addWidget(rightPanel);
    splitter->setStretchFactor(0, 0);
    splitter->setStretchFactor(1, 1);

    mainLayout->addWidget(splitter);
    setCentralWidget(centralWidget);

    applyIntroAnimation(header, 0);
    applyIntroAnimation(controlGroup, 90);
    applyIntroAnimation(perfGroup, 160);
    applyIntroAnimation(imageCard, 120);
}

void MainWindow::createImageCard(QVBoxLayout* layout, QLabel*& imgLbl, QLabel*& timeLbl, const QString& title, const QString& subtitle) {
    QLabel* titleLbl = new QLabel(title);
    titleLbl->setAlignment(Qt::AlignCenter);
    titleLbl->setStyleSheet("font-weight: 800; font-size: 15px; color: #1f2937;");

    QLabel* subLbl = new QLabel(subtitle);
    subLbl->setAlignment(Qt::AlignCenter);
    subLbl->setStyleSheet("color: #64748b; font-size: 12px;");

    imgLbl = new QLabel("Load an image to preview segmentation");
    imgLbl->setAlignment(Qt::AlignCenter);
    imgLbl->setMinimumSize(320, 260);
    imgLbl->setStyleSheet("border: 2px dashed #c7d2fe; background-color: #f8fafc; color: #475569; font-size: 13px; border-radius: 12px;");

    timeLbl = new QLabel("Time: -- ms");
    timeLbl->setAlignment(Qt::AlignCenter);
    timeLbl->setStyleSheet("font-weight: 700; font-size: 13px; color: #0f766e; background-color: transparent;");

    layout->addWidget(titleLbl);
    layout->addWidget(subLbl);
    layout->addWidget(imgLbl);
    layout->addWidget(timeLbl);
}

QPixmap MainWindow::createPerformanceChart(int cpuMs, int gpuMs) {
    QPixmap pix(360, 170);
    pix.fill(Qt::transparent);

    QPainter painter(&pix);
    painter.setRenderHint(QPainter::Antialiasing, true);

    QLinearGradient bg(0, 0, 0, pix.height());
    bg.setColorAt(0.0, QColor("#ffffff"));
    bg.setColorAt(1.0, QColor("#f3f6fb"));
    painter.fillRect(pix.rect(), bg);

    painter.setPen(QPen(QColor("#e6ecf4"), 1));
    for (int y = 32; y <= 140; y += 27) {
        painter.drawLine(40, y, 330, y);
    }

    int maxValue = 1;
    if (cpuMs > maxValue) maxValue = cpuMs;
    if (gpuMs > maxValue) maxValue = gpuMs;

    auto barHeight = [&](int value) {
        if (value <= 0) return 10;
        return static_cast<int>(92.0 * value / maxValue) + 10;
    };

    int cpuHeight = barHeight(cpuMs);
    int gpuHeight = barHeight(gpuMs);

    QRect cpuBar(90, 140 - cpuHeight, 70, cpuHeight);
    QRect gpuBar(200, 140 - gpuHeight, 70, gpuHeight);

    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor("#f59e0b"));
    painter.drawRoundedRect(cpuBar, 6, 6);
    painter.setBrush(QColor("#14b8a6"));
    painter.drawRoundedRect(gpuBar, 6, 6);

    painter.setPen(QColor("#3b4252"));
    painter.drawText(QRect(80, 145, 90, 20), Qt::AlignCenter, "CPU");
    painter.drawText(QRect(190, 145, 90, 20), Qt::AlignCenter, "GPU");

    painter.setPen(QColor("#5b6475"));
    painter.drawText(QRect(10, 8, 180, 18), Qt::AlignLeft, "Runtime (ms)");

    if (cpuMs > 0) {
        painter.drawText(cpuBar.adjusted(0, -20, 0, -2), Qt::AlignCenter, QString::number(cpuMs));
    }
    if (gpuMs > 0) {
        painter.drawText(gpuBar.adjusted(0, -20, 0, -2), Qt::AlignCenter, QString::number(gpuMs));
    }

    return pix;
}

void MainWindow::applyIntroAnimation(QWidget* widget, int delayMs) {
    if (!widget) return;

    auto *effect = new QGraphicsOpacityEffect(widget);
    effect->setOpacity(0.0);
    widget->setGraphicsEffect(effect);

    auto *anim = new QPropertyAnimation(effect, "opacity", this);
    anim->setStartValue(0.0);
    anim->setEndValue(1.0);
    anim->setDuration(520);
    anim->setEasingCurve(QEasingCurve::OutCubic);
    introAnimations.append(anim);

    QTimer::singleShot(delayMs, anim, [anim]() { anim->start(); });
}


void MainWindow::connectSignals() {
    // Route button clicks to their respective class methods
    connect(btnLoad, &QPushButton::clicked, this, &MainWindow::loadImage);
    
    connect(btnRunCPU, &QPushButton::clicked, this, [this]() { 
        runSegmentation(false); 
    });
    
    connect(btnRunGPU, &QPushButton::clicked, this, [this]() { 
        runSegmentation(true); 
    });
}

void MainWindow::loadImage() {
    QString exePath = QCoreApplication::applicationDirPath();
    QString relativePath = exePath + "/../../resources/brats dataset/";
    QString fileName = QFileDialog::getOpenFileName(this, "Open Image", relativePath, "Images (*.png *.jpg *.jpeg *.bmp)");
    if (fileName.isEmpty()) return;

    originalImage.load(fileName);
    originalImage = originalImage.convertToFormat(QImage::Format_RGB888); // Ensure 24-bit RGB
    
    lastImageSize = originalImage.size();
    QPixmap pixmap = QPixmap::fromImage(originalImage).scaled(420, 320, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    lblImgOrig->setPixmap(pixmap);
    lblImgCPU->setPixmap(pixmap);
    lblImgGPU->setPixmap(pixmap);
    
    // Reset labels and stats
    lblStatus->setText("Image loaded");
    resetPerformanceUI();
    
    // Enable processing buttons now that we have data
    btnRunCPU->setEnabled(true);
    btnRunGPU->setEnabled(true);
}

void MainWindow::runSegmentation(bool useGPU) {
    if (originalImage.isNull()) return;

    // 1. Prepare Data
    QImage processImage = originalImage.copy();
    int methodIndex = comboMethod->currentIndex();
    int k = spinK->value();
    unsigned char* ptr = processImage.bits();
    int w = processImage.width();
    int h = processImage.height();

    // 2. Start Timer
    QElapsedTimer timer;
    timer.start();
    
    // 3. Dispatch to Algorithm
    if(useGPU){
    switch (methodIndex) {
        case 0: runBasicKMeans(ptr, w, h, 3, k, useGPU); break;
        case 1: runTiledKMeans(ptr, w, h, 3, k); break;
        case 2: runFuzzyCMeans(ptr, w, h, 3, k); break;
        case 3: runMiniBatchKMeans(ptr, w, h, 3, k); break;
        default: QMessageBox::warning(this, "Error", "Method not implemented."); return;
    }
}
    else{

        runBasicKMeans(ptr, w, h, 3, k, useGPU);

    }
    // 4. Update UI
    int elapsedMs = static_cast<int>(timer.elapsed());
    QString timeText = QString("Time: %1 ms").arg(elapsedMs);
    QPixmap resultPixmap = QPixmap::fromImage(processImage).scaled(420, 320, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    if (useGPU) {
        lblTimeGPU->setText(timeText);
        lblImgGPU->setPixmap(resultPixmap);
    } else {
        lblTimeCPU->setText(timeText);
        lblImgCPU->setPixmap(resultPixmap);
    }

    updatePerformanceUI(useGPU, elapsedMs);
}

void MainWindow::resetPerformanceUI() {
    lastCpuMs = -1;
    lastGpuMs = -1;
    lblTimeCPU->setText("Time: -- ms");
    lblTimeGPU->setText("Time: -- ms");
    tblStats->item(0, 1)->setText("--");
    tblStats->item(1, 1)->setText("--");
    tblStats->item(2, 1)->setText("--");
    tblStats->item(3, 1)->setText(comboMethod->currentText());
    lblChart->setPixmap(createPerformanceChart(lastCpuMs, lastGpuMs));
}

void MainWindow::updatePerformanceUI(bool useGPU, int elapsedMs) {
    if (useGPU) {
        lastGpuMs = elapsedMs;
    } else {
        lastCpuMs = elapsedMs;
    }

    if (lastGpuMs > 0) {
        tblStats->item(0, 1)->setText(QString("%1 ms").arg(lastGpuMs));
    }

    if (lastCpuMs > 0 && lastGpuMs > 0) {
        double speedup = static_cast<double>(lastCpuMs) / static_cast<double>(lastGpuMs);
        tblStats->item(1, 1)->setText(QString("%1x").arg(speedup, 0, 'f', 2));
    } else {
        tblStats->item(1, 1)->setText("--");
    }

    if (!lastImageSize.isEmpty()) {
        tblStats->item(2, 1)->setText(QString("%1 x %2").arg(lastImageSize.width()).arg(lastImageSize.height()));
    }

    tblStats->item(3, 1)->setText(comboMethod->currentText());
    lblChart->setPixmap(createPerformanceChart(lastCpuMs, lastGpuMs));
    lblStatus->setText(useGPU ? "GPU run complete" : "CPU run complete");
}

// =====================================================================
// ENTRY POINT
// =====================================================================
int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    MainWindow window;
    window.show();
    return app.exec();
}