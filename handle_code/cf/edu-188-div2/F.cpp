#include <bits/stdc++.h>
using namespace std;

static const int MOD = 998244353;

int addmod(int a, int b) {
    int s = a + b;
    if (s >= MOD) s -= MOD;
    return s;
}

int submod(int a, int b) {
    int s = a - b;
    if (s < 0) s += MOD;
    return s;
}

int mulmod(long long a, long long b) {
    return int((a * b) % MOD);
}

int qpow(int a, long long e) {
    int r = 1;
    while (e) {
        if (e & 1) r = mulmod(r, a);
        a = mulmod(a, a);
        e >>= 1;
    }
    return r;
}

struct InvCache {
    unordered_map<int, int> mp;
    int get(int x) {
        auto it = mp.find(x);
        if (it != mp.end()) return it->second;
        int v = qpow(x, MOD - 2);
        mp.emplace(x, v);
        return v;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    cin >> n >> m;
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    vector<int> ks(m);
    for (int i = 0; i < m; i++) cin >> ks[i];

    // w_i = count of subarrays where i is the chosen minimum representative.
    vector<int> L(n + 1), R(n + 1), st;
    st.reserve(n);

    // previous strictly smaller (<)
    for (int i = 1; i <= n; i++) {
        while (!st.empty() && a[st.back()] >= a[i]) st.pop_back();
        L[i] = st.empty() ? 0 : st.back();
        st.push_back(i);
    }

    st.clear();
    // next smaller or equal (<=)
    for (int i = n; i >= 1; i--) {
        while (!st.empty() && a[st.back()] > a[i]) st.pop_back();
        R[i] = st.empty() ? (n + 1) : st.back();
        st.push_back(i);
    }

    vector<pair<int, int>> vw;
    vw.reserve(n);
    for (int i = 1; i <= n; i++) {
        long long w = 1LL * (i - L[i]) * (R[i] - i) % MOD;
        vw.push_back({a[i], (int)w});
    }
    sort(vw.begin(), vw.end());

    vector<int> vals, W;
    vals.reserve(n);
    W.reserve(n);
    for (auto [v, w] : vw) {
        if (!vals.empty() && vals.back() == v) {
            W.back() = addmod(W.back(), w);
        } else {
            vals.push_back(v);
            W.push_back(w);
        }
    }
    int U = (int)vals.size();

    InvCache inv_cache;

    // C = sum over all subarrays of sum(1 / a_i) = sum_i inv(a_i) * i * (n-i+1)
    int C = 0;
    for (int i = 1; i <= n; i++) {
        int inva = inv_cache.get(a[i]);
        long long occ = 1LL * i * (n - i + 1) % MOD;
        C = addmod(C, mulmod(occ, inva));
    }

    // Prefix sums on value groups:
    // prefW: sum W
    // prefWV: sum W * v
    // prefWInv: sum W / v
    vector<int> prefW(U + 1, 0), prefWV(U + 1, 0), prefWInv(U + 1, 0);
    for (int i = 0; i < U; i++) {
        int invv = inv_cache.get(vals[i]);
        prefW[i + 1] = addmod(prefW[i], W[i]);
        prefWV[i + 1] = addmod(prefWV[i], mulmod(W[i], vals[i] % MOD));
        prefWInv[i + 1] = addmod(prefWInv[i], mulmod(W[i], invv));
    }
    int totalWInv = prefWInv[U];

    // For each k:
    // sum_v W[v] * g(v, k)
    // g(v,k)=k/v (k<v), else k-v+2-1/v
    // Let LE={v<=k}, GT={v>k}
    // = (k+2)*sumW_LE - sum(Wv)_LE - sum(W/v)_LE + k*sum(W/v)_GT
    // = (k+2)*S0 - S1 + k*T - (k+1)*S2
    // where S0=prefW, S1=prefWV, S2=prefWInv, T=totalWInv.
    for (int i = 0; i < m; i++) {
        int k = ks[i];
        int pos = upper_bound(vals.begin(), vals.end(), k) - vals.begin();

        int S0 = prefW[pos];
        int S1 = prefWV[pos];
        int S2 = prefWInv[pos];

        int km = k % MOD;
        int term1 = mulmod((km + 2) % MOD, S0);
        int term2 = S1;
        int term3 = mulmod(km, totalWInv);
        int term4 = mulmod((km + 1) % MOD, S2);

        int add = submod(addmod(term1, term3), addmod(term2, term4));
        int ans = addmod(C, add);
        cout << ans << '\n';
    }

    return 0;
}

