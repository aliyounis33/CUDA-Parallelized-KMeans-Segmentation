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
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);
    mainLayout->setSpacing(16);
    mainLayout->setContentsMargins(20, 20, 20, 20);

    centralWidget->setStyleSheet(
        "QWidget { background-color: #f4f6f9; color: #1d2433; font-family: 'Segoe UI', 'Arial'; }"
        "QGroupBox { border: 1px solid #d7dce5; border-radius: 10px; margin-top: 12px; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 2px 6px; color: #1d2433; font-weight: 600; }"
        "QFrame#card { background: #ffffff; border: 1px solid #e1e6ef; border-radius: 12px; }"
        "QPushButton { background: #0f62fe; color: #ffffff; border: none; border-radius: 8px; padding: 8px 14px; font-weight: 600; }"
        "QPushButton:disabled { background: #a9b5d1; }"
        "QPushButton#ghost { background: #e8eefc; color: #0f62fe; border: 1px solid #c6d4fb; }"
        "QComboBox, QSpinBox { background: #ffffff; border: 1px solid #cfd6e6; border-radius: 6px; padding: 6px 8px; }"
        "QLabel#title { font-size: 22px; font-weight: 700; }"
        "QLabel#subtitle { color: #5b6475; }"
        "QLabel#badge { background: #e7f5ff; color: #0a4c7d; border-radius: 8px; padding: 4px 8px; font-weight: 600; }"
        "QTableWidget { background: #ffffff; border: 1px solid #e1e6ef; border-radius: 8px; gridline-color: #eef1f6; }"
        "QHeaderView::section { background: #f1f4f9; padding: 6px; border: none; font-weight: 600; }"
    );

    // --- 1. Header ---
    QFrame *header = new QFrame();
    header->setObjectName("card");
    QHBoxLayout *headerLayout = new QHBoxLayout(header);
    headerLayout->setContentsMargins(16, 12, 16, 12);

    QVBoxLayout *headerTextLayout = new QVBoxLayout();
    QLabel *title = new QLabel("CUDA K-Means Segmentation Profiler");
    title->setObjectName("title");
    QLabel *subtitle = new QLabel("Compare CPU vs GPU segmentation with live previews and performance analytics.");
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
    QGridLayout *controlLayout = new QGridLayout(controlGroup);
    controlLayout->setHorizontalSpacing(10);
    controlLayout->setVerticalSpacing(10);

    btnLoad = new QPushButton("Load Image");
    btnLoad->setObjectName("ghost");

    comboMethod = new QComboBox();
    comboMethod->addItems({"Basic K-Means", "Tiled K-Means", "Fuzzy C-Means", "Parallel K-Means++", "Mini-Batch K-Means"});

    spinK = new QSpinBox();
    spinK->setRange(2, 32);
    spinK->setValue(3);

    btnRunCPU = new QPushButton("Run on CPU");
    btnRunGPU = new QPushButton("Run on GPU");
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
    QVBoxLayout *perfLayout = new QVBoxLayout(perfGroup);
    tblStats = new QTableWidget(5, 2);
    tblStats->setHorizontalHeaderLabels({"Metric", "Value"});
    tblStats->verticalHeader()->setVisible(false);
    tblStats->setEditTriggers(QAbstractItemView::NoEditTriggers);
    tblStats->setSelectionMode(QAbstractItemView::NoSelection);
    tblStats->horizontalHeader()->setStretchLastSection(true);
    tblStats->setRowHeight(0, 26);
    tblStats->setRowHeight(1, 26);
    tblStats->setRowHeight(2, 26);
    tblStats->setRowHeight(3, 26);
    tblStats->setRowHeight(4, 26);

    tblStats->setItem(0, 0, new QTableWidgetItem("CPU time"));
    tblStats->setItem(1, 0, new QTableWidgetItem("GPU time"));
    tblStats->setItem(2, 0, new QTableWidgetItem("Speedup"));
    tblStats->setItem(3, 0, new QTableWidgetItem("Image size"));
    tblStats->setItem(4, 0, new QTableWidgetItem("Method"));

    for (int row = 0; row < 5; ++row) {
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
}

void MainWindow::createImageCard(QVBoxLayout* layout, QLabel*& imgLbl, QLabel*& timeLbl, const QString& title, const QString& subtitle) {
    QLabel* titleLbl = new QLabel(title);
    titleLbl->setAlignment(Qt::AlignCenter);
    titleLbl->setStyleSheet("font-weight: 700; font-size: 15px;");

    QLabel* subLbl = new QLabel(subtitle);
    subLbl->setAlignment(Qt::AlignCenter);
    subLbl->setStyleSheet("color: #6b7280; font-size: 12px;");

    imgLbl = new QLabel("Load an image to preview segmentation");
    imgLbl->setAlignment(Qt::AlignCenter);
    imgLbl->setMinimumSize(320, 260);
    imgLbl->setStyleSheet("border: 2px dashed #c8d0de; background-color: #f8fafc; color: #4b5563; font-size: 13px;");

    timeLbl = new QLabel("Time: -- ms");
    timeLbl->setAlignment(Qt::AlignCenter);
    timeLbl->setStyleSheet("font-weight: 600; font-size: 13px; color: #0f62fe; background-color: transparent;");

    layout->addWidget(titleLbl);
    layout->addWidget(subLbl);
    layout->addWidget(imgLbl);
    layout->addWidget(timeLbl);
}

QPixmap MainWindow::createPerformanceChart(int cpuMs, int gpuMs) {
    QPixmap pix(360, 160);
    pix.fill(Qt::transparent);

    QPainter painter(&pix);
    painter.setRenderHint(QPainter::Antialiasing, true);

    QLinearGradient bg(0, 0, 0, pix.height());
    bg.setColorAt(0.0, QColor("#ffffff"));
    bg.setColorAt(1.0, QColor("#f4f7fb"));
    painter.fillRect(pix.rect(), bg);

    painter.setPen(QPen(QColor("#e1e6ef"), 1));
    for (int y = 30; y <= 130; y += 25) {
        painter.drawLine(40, y, 330, y);
    }

    int maxValue = 1;
    if (cpuMs > maxValue) maxValue = cpuMs;
    if (gpuMs > maxValue) maxValue = gpuMs;

    auto barHeight = [&](int value) {
        if (value <= 0) return 10;
        return static_cast<int>(90.0 * value / maxValue) + 10;
    };

    int cpuHeight = barHeight(cpuMs);
    int gpuHeight = barHeight(gpuMs);

    QRect cpuBar(90, 130 - cpuHeight, 70, cpuHeight);
    QRect gpuBar(200, 130 - gpuHeight, 70, gpuHeight);

    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor("#6fa8ff"));
    painter.drawRoundedRect(cpuBar, 6, 6);
    painter.setBrush(QColor("#23c58e"));
    painter.drawRoundedRect(gpuBar, 6, 6);

    painter.setPen(QColor("#3b4252"));
    painter.drawText(QRect(80, 135, 90, 20), Qt::AlignCenter, "CPU");
    painter.drawText(QRect(190, 135, 90, 20), Qt::AlignCenter, "GPU");

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
        case 3: runKMeansPlusPlus(ptr, w, h, 3, k); break;
        case 4: runMiniBatchKMeans(ptr, w, h, 3, k); break;
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
    tblStats->item(3, 1)->setText("--");
    tblStats->item(4, 1)->setText(comboMethod->currentText());
    lblChart->setPixmap(createPerformanceChart(lastCpuMs, lastGpuMs));
}

void MainWindow::updatePerformanceUI(bool useGPU, int elapsedMs) {
    if (useGPU) {
        lastGpuMs = elapsedMs;
    } else {
        lastCpuMs = elapsedMs;
    }

    if (lastCpuMs > 0) {
        tblStats->item(0, 1)->setText(QString("%1 ms").arg(lastCpuMs));
    }
    if (lastGpuMs > 0) {
        tblStats->item(1, 1)->setText(QString("%1 ms").arg(lastGpuMs));
    }

    if (lastCpuMs > 0 && lastGpuMs > 0) {
        double speedup = static_cast<double>(lastCpuMs) / static_cast<double>(lastGpuMs);
        tblStats->item(2, 1)->setText(QString("%1x").arg(speedup, 0, 'f', 2));
    } else {
        tblStats->item(2, 1)->setText("--");
    }

    if (!lastImageSize.isEmpty()) {
        tblStats->item(3, 1)->setText(QString("%1 x %2").arg(lastImageSize.width()).arg(lastImageSize.height()));
    }

    tblStats->item(4, 1)->setText(comboMethod->currentText());
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