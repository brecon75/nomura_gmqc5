#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <memory>

using namespace std;

double parseDays(const string& s) {
    char unit = s.back();
    double v  = stod(s.substr(0, s.size() - 1));
    if (unit == 'D' || unit == 'd') return v;
    if (unit == 'W' || unit == 'w') return v * 7;
    if (unit == 'M' || unit == 'm') return v * 30;
    if (unit == 'Y' || unit == 'y') return v * 360;
    return v;
}

inline double dcf(double t) { return t / 360.0; }


// Interpolator
// Sparse log-DF weights: ln DF(t) = sum_j  w_j * ln DF[j]
using Weights = vector<pair<int, double>>;

class Interpolator {
public:
    virtual ~Interpolator() = default;
    virtual double  interpolate(double t, const vector<double>& T,
                                const vector<double>& DF) const = 0;
    virtual Weights getWeights (double t, const vector<double>& T) const = 0;
};

class LinearInterpolator : public Interpolator {
public:
    double interpolate(double t, const vector<double>& T,
                       const vector<double>& DF) const override {
        if (t == 0.0) return 1.0;
        int n = (int)T.size();
        for (int i = 0; i < n; ++i)
            if (abs(t - T[i]) < 1e-9) return DF[i];

        int idx = 0;
        while (idx < n && T[idx] < t) ++idx;
        if (idx == n) idx = n - 1;

        double t0  = (idx == 0) ? 0.0   : T[idx-1];
        double df0 = (idx == 0) ? 1.0   : DF[idx-1];
        double w1  = (t - t0) / (T[idx] - t0);
        return exp((1-w1)*log(df0) + w1*log(DF[idx]));
    }

    Weights getWeights(double t, const vector<double>& T) const override {
        Weights res;
        if (t == 0.0) return res;
        int n = (int)T.size();
        for (int i = 0; i < n; ++i)
            if (abs(t - T[i]) < 1e-9) { res.push_back({i, 1.0}); return res; }

        int idx = 0;
        while (idx < n && T[idx] < t) ++idx;
        if (idx == n) idx = n - 1;

        double t0 = (idx == 0) ? 0.0 : T[idx-1];
        double w1 = (t - t0) / (T[idx] - t0);
        if (idx > 0) res.push_back({idx-1, 1.0-w1});
        res.push_back({idx, w1});
        return res;
    }
};

class AQInterpolator : public Interpolator {
public:
    double interpolate(double t, const vector<double>& T,
                       const vector<double>& DF) const override {
        if (t == 0.0) return 1.0;
        double lndf = 0.0;
        for (auto [j, w] : getWeights(t, T))
            lndf += w * log(DF[j]);
        return exp(lndf);
    }

    Weights getWeights(double t, const vector<double>& T) const override {
        Weights res;
        if (t == 0.0) return res;
        int N = (int)T.size() - 1;

        for (int i = 0; i <= N; ++i)
            if (abs(t - T[i]) < 1e-9) { if (i > 0) res.push_back({i, 1.0}); return res; }

        int k = 1;
        while (k <= N && T[k] < t) ++k;
        if (k > N) k = N;

        if (k == 1) {
            // First interval: linear by spec
            double w = (t - T[0]) / (T[1] - T[0]);
            res.push_back({1, w});
        } else if (k == N) {
            // Last interval: single quadratic Q_{N-1}
            addQuadraticWeights(res, k-1, t, T, 1.0);
        } else {
            // Interior: blend Q_{k-1} and Q_k
            double alpha = (T[k] - t)   / (T[k] - T[k-1]);
            double beta  = (t - T[k-1]) / (T[k] - T[k-1]);
            addQuadraticWeights(res, k-1, t, T, alpha);
            addQuadraticWeights(res, k,   t, T, beta);
        }
        return res;
    }

private:
    // Accumulate weights for quadratic Q_q (centred at node q) into res,
    // scaled by blendWeight.
    void addQuadraticWeights(Weights& res, int q, double t,
                             const vector<double>& T,
                             double blendWeight) const {
        map<int,double> tmp;
        for (int a : {q-1, q, q+1})
            tmp[a] += blendWeight * lagrange(q, a, t, T);
        for (auto [idx, w] : tmp)
            if (idx > 0 && abs(w) > 1e-12)
                res.push_back({idx, w});
    }

    double lagrange(int q, int a, double t, const vector<double>& T) const {
        double num = 1.0, den = 1.0;
        for (int j : {q-1, q, q+1})
            if (j != a) { num *= (t - T[j]); den *= (T[a] - T[j]); }
        return num / den;
    }
};



// Instrument

// Each instrument pins one node of the discount curve.  It exposes:
//   - a direct closed-form DF (hasClosedForm=true), or
//   - a residual F(DF_m)=0 for Newton, plus its partial derivatives.
// Add a new instrument by subclassing and implementing the five methods.

class Instrument {
public:
    virtual ~Instrument() = default;
    virtual double maturityDays()  const = 0;
    virtual double quote()         const = 0;
    virtual bool   hasClosedForm() const = 0;
    virtual double closedFormDF()       const = 0;
    virtual double closedFormDFDeriv()  const = 0;
    virtual double residual    (const vector<double>& T, const vector<double>& DF, const Interpolator& interp) const = 0;
    virtual double dRes_dDFm   (const vector<double>& T, const vector<double>& DF, const Interpolator& interp) const = 0;
    virtual vector<double> dRes_dDFl(const vector<double>& T, const vector<double>& DF, const Interpolator& interp) const = 0;
    virtual double dRes_dQuote (const vector<double>& T, const vector<double>& DF, const Interpolator& interp) const = 0;
};

class CashDeposit : public Instrument {
public:
    CashDeposit(double t, double rate) : t_(t), r_(rate) {}
    double maturityDays()  const override { return t_; }
    double quote()         const override { return r_; }
    bool   hasClosedForm() const override { return true; }

    double closedFormDF()      const override { return 1.0 / (1.0 + r_*t_/360.0); }
    double closedFormDFDeriv() const override {
        double d = 1.0 + r_*t_/360.0;
        return -(t_/360.0) / (d*d);
    }

    // Residual path provided for completeness; not called when hasClosedForm=true.
    double residual(const vector<double>&, const vector<double>& DF,
                    const Interpolator&) const override {
        return DF.back() * (1.0 + r_*t_/360.0) - 1.0;
    }
    double dRes_dDFm(const vector<double>&, const vector<double>&,
                     const Interpolator&) const override { return 1.0 + r_*t_/360.0; }
    vector<double> dRes_dDFl(const vector<double>&,
                                   const vector<double>& DF,
                                   const Interpolator&) const override {
        return vector<double>(DF.size(), 0.0);
    }
    double dRes_dQuote(const vector<double>&, const vector<double>& DF,
                       const Interpolator&) const override { return DF.back() * t_/360.0; }

private:
    double t_, r_;
};

class VanillaSwap : public Instrument {
public:
    VanillaSwap(double t, double rate, double freq = 180.0)
        : t_(t), r_(rate), freq_(freq), K_((int)round(t/freq)) {}

    double maturityDays()  const override { return t_; }
    double quote()         const override { return r_; }
    bool   hasClosedForm() const override { return t_ <= freq_; }

    double closedFormDF()      const override { return 1.0 / (1.0 + r_*t_/360.0); }
    double closedFormDFDeriv() const override {
        double d = 1.0 + r_*t_/360.0;
        return -(t_/360.0) / (d*d);
    }

    double residual(const vector<double>& T, const vector<double>& DF,
                    const Interpolator& interp) const override {
        double F = 1.0 - DF.back();
        for (int k = 1; k <= K_; ++k)
            F -= r_ * interp.interpolate(k*freq_, T, DF) * dcf(freq_);
        return F;
    }

    double dRes_dDFm(const vector<double>& T, const vector<double>& DF,
                     const Interpolator& interp) const override {
        int m = (int)DF.size() - 1;
        double d = -1.0;
        for (int k = 1; k <= K_; ++k) {
            double tk = k*freq_, df_tk = interp.interpolate(tk, T, DF);
            for (auto [j, w] : interp.getWeights(tk, T))
                if (j == m) d -= r_ * dcf(freq_) * df_tk * w / DF[m];
        }
        return d;
    }

    vector<double> dRes_dDFl(const vector<double>& T, const vector<double>& DF,
                                   const Interpolator& interp) const override {
        int m = (int)DF.size() - 1;
        vector<double> d(m+1, 0.0);
        d[m] = -1.0;
        for (int k = 1; k <= K_; ++k) {
            double tk = k*freq_, df_tk = interp.interpolate(tk, T, DF);
            for (auto [j, w] : interp.getWeights(tk, T))
                d[j] -= r_ * dcf(freq_) * df_tk * w / DF[j];
        }
        return d;
    }

    double dRes_dQuote(const vector<double>& T, const vector<double>& DF,
                       const Interpolator& interp) const override {
        double d = 0.0;
        for (int k = 1; k <= K_; ++k)
            d -= interp.interpolate(k*freq_, T, DF) * dcf(freq_);
        return d;
    }

private:
    double t_, r_, freq_;
    int K_;
};


// DiscountCurve
struct DiscountCurve {
    vector<double> T, DF;
    vector<vector<double>> J; // J[m][i] = d DF_m / d quote_i

    double df(double t, const Interpolator& interp) const {
        return interp.interpolate(t, T, DF);
    }
    int n() const { return (int)T.size() - 1; }
};


// Calibration: bootstrap any ordered list of Instruments
DiscountCurve calibrate(const vector<unique_ptr<Instrument>>& instruments,
                         const Interpolator& interp,
                         int maxIter = 100, double tol = 1e-12) {
    int n = (int)instruments.size();
    DiscountCurve curve;
    curve.T.assign(n+1, 0.0);
    curve.DF.assign(n+1, 1.0);
    curve.J.assign(n+1, vector<double>(n+1, 0.0));

    for (int m = 1; m <= n; ++m) {
        const Instrument& inst = *instruments[m-1];
        curve.T[m] = inst.maturityDays();

        if (inst.hasClosedForm()) {
            curve.DF[m]  = inst.closedFormDF();
            curve.J[m][m] = inst.closedFormDFDeriv();
            continue;
        }

        // Working curve prefix up to node m
        vector<double> cT(curve.T.begin(),  curve.T.begin()  + m+1);
        vector<double> cDF(curve.DF.begin(), curve.DF.begin() + m+1);
        cDF[m] = exp(-inst.quote() * inst.maturityDays() / 360.0);

        // Newton-Raphson
        for (int iter = 0; iter < maxIter; ++iter) {
            double F = inst.residual(cT, cDF, interp);
            if (abs(F) < tol) break;
            double dF = inst.dRes_dDFm(cT, cDF, interp);
            if (abs(dF) < 1e-15) throw runtime_error("Newton diverged at node " + to_string(m));
            cDF[m] -= F / dF;
            if (iter == maxIter-1) throw runtime_error("Newton did not converge at node " + to_string(m));
        }
        curve.DF[m] = cDF[m];

        // Jacobian via implicit differentiation:
        // J[m][i] = -(dF/dp_i + sum_{l<m} dF/dDF_l * J[l][i]) / dF/dDF_m
        auto   dFl = inst.dRes_dDFl(cT, cDF, interp);
        double dFm = inst.dRes_dDFm(cT, cDF, interp);
        double dFp = inst.dRes_dQuote(cT, cDF, interp);

        for (int i = 1; i <= m; ++i) {
            double chain = (i == m) ? dFp : 0.0;
            for (int l = 1; l < m; ++l)
                chain += dFl[l] * curve.J[l][i];
            curve.J[m][i] = -chain / dFm;
        }
    }
    return curve;
}


// Pricing: PV, par swap rate, and analytic risk for a new swap
struct SwapSpec {
    double notional, fixedRate, maturityDays, fixedFreqDays;
};

struct PricingResult {
    double pv, parSwapRate;
    vector<double> risk;
};

PricingResult priceSwap(const SwapSpec& s, const DiscountCurve& curve,
                         const Interpolator& interp) {
    int n = curve.n();
    int Kfix = (int)round(s.maturityDays / s.fixedFreqDays);
    double cf = dcf(s.fixedFreqDays);

    double pvFloat = s.notional * (1.0 - curve.df(s.maturityDays, interp));
    double annuity = 0.0;
    for (int k = 1; k <= Kfix; ++k)
        annuity += curve.df(k * s.fixedFreqDays, interp) * cf;

    PricingResult res;
    res.pv          = pvFloat - s.notional * s.fixedRate * annuity;
    res.parSwapRate  = pvFloat / (s.notional * annuity) * 100.0;

    // d PV / d DF_j  via log-space chain rule: d DF(t)/d DF_j = DF(t)*w_j/DF_j
    vector<double> dPV(n+1, 0.0);
    auto accumulate = [&](double t, double scale) {
        double df_t = curve.df(t, interp);
        for (auto [j, w] : interp.getWeights(t, curve.T))
            dPV[j] += scale * df_t * w / curve.DF[j];
    };
    accumulate(s.maturityDays, -s.notional);
    for (int k = 1; k <= Kfix; ++k)
        accumulate(k * s.fixedFreqDays, -s.notional * s.fixedRate * cf);

    // risk[i] = d PV / d quote_i = sum_m (d PV/d DF_m) * J[m][i]
    res.risk.assign(n, 0.0);
    for (int i = 1; i <= n; ++i)
        for (int m = 1; m <= n; ++m)
            res.risk[i-1] += dPV[m] * curve.J[m][i];

    return res;
}

// Input / Output
struct MarketRow { double t, cashRate, swapRate; };

vector<MarketRow> readMarketData(ifstream& f, int n) {
    vector<MarketRow> rows(n);
    string line, tok;
    for (int i = 0; i < n; ++i) {
        getline(f, line);
        stringstream ss(line);
        getline(ss, tok, ','); rows[i].t        = parseDays(tok);
        getline(ss, tok, ','); rows[i].cashRate  = stod(tok) / 100.0;
        getline(ss, tok, ','); rows[i].swapRate  = stod(tok) / 100.0;
    }
    return rows;
}



int main() {
    ifstream f("Input.csv");
    if (!f.is_open()) { cerr << "Cannot open Input.csv\n"; return 1; }

    string line, tok;
    getline(f, line);

    if (line.size() >= 3 && 
        (unsigned char)line[0] == 0xEF && 
        (unsigned char)line[1] == 0xBB && 
        (unsigned char)line[2] == 0xBF) {
        line = line.substr(3);
    }
    int numNodes = stoi(line.substr(0, line.find(',')));

    auto mkt = readMarketData(f, numNodes);

    getline(f, line);
    double t_q1 = stod(line.substr(0, line.find(',')));

    getline(f, line);
    stringstream ss(line);
    SwapSpec newSwap; newSwap.notional = 100.0;
    getline(ss, tok, ','); newSwap.fixedRate     = stod(tok) / 100.0;
    getline(ss, tok, ','); newSwap.maturityDays  = parseDays(tok);
    getline(ss, tok, ','); newSwap.fixedFreqDays = parseDays(tok);
    f.close();

    // Build instrument lists
    auto makeCash = [&]() {
        vector<unique_ptr<Instrument>> v;
        for (auto& r : mkt) v.push_back(make_unique<CashDeposit>(r.t, r.cashRate));
        return v;
    };
    auto makeSwap = [&]() {
        vector<unique_ptr<Instrument>> v;
        for (auto& r : mkt) v.push_back(make_unique<VanillaSwap>(r.t, r.swapRate));
        return v;
    };

    LinearInterpolator lin;
    AQInterpolator     aq;

    DiscountCurve cashLin = calibrate(makeCash(), lin);
    DiscountCurve cashAQ  = calibrate(makeCash(), aq);
    DiscountCurve swapLin = calibrate(makeSwap(), lin);
    DiscountCurve swapAQ  = calibrate(makeSwap(), aq);

    auto rCL = priceSwap(newSwap, cashLin, lin);
    auto rCA = priceSwap(newSwap, cashAQ,  aq);
    auto rSL = priceSwap(newSwap, swapLin, lin);
    auto rSA = priceSwap(newSwap, swapAQ,  aq);

    ofstream out("Output.csv");
    out << fixed << setprecision(10);

    //"Q1.a) for Cash/Linear,Q1.b) for Cash/AQ,Q1.c) for Swap/Linear,Q1.d) for Swap/AQ\n";
    out << lin.interpolate(t_q1, cashLin.T, cashLin.DF) << ","
        << aq .interpolate(t_q1, cashAQ .T, cashAQ .DF) << ","
        << lin.interpolate(t_q1, swapLin.T, swapLin.DF) << ","
        << aq .interpolate(t_q1, swapAQ .T, swapAQ .DF) << "\n";

    //"Q2.1.a)for Cash/Linear,Q2.1.b) for Cash/AQ,Q2.1.c) for Swap/Linear,Q2.1.d) for Swap/AQ\n";
    out << rCL.pv << "," << rCA.pv << "," << rSL.pv << "," << rSA.pv << "\n";

    // out << "Q2.1.a)Par-Swap Rate for Cash/Linear,Q2.1.b)Par-Swap Rate for Cash/AQ,"
    //        "Q2.1.c)Par-Swap Rate for Swap/Linear,Q2.1.d)Par-Swap Rate for Swap/AQ\n";
    out << rCL.parSwapRate << "," << rCA.parSwapRate << "," << rSL.parSwapRate << "," << rSA.parSwapRate << "\n";

    //"Q2.2.a) for Cash/Linear,Q2.2.b) for Cash/AQ,Q2.2.c) for Swap/Linear,Q2.2.d) for Swap/AQ\n";
    for (int i = 0; i < numNodes; ++i)
        out << rCL.risk[i] << "," << rCA.risk[i] << "," << rSL.risk[i] << "," << rSA.risk[i] << "\n";

    out.close();
    cout << "Results written to Output.csv\n";
    return 0;
}