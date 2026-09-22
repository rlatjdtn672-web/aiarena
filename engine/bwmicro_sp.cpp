// bwmicro.cpp — 강화학습용 스타크래프트 전투 환경 (유닛 조합을 인자로 받는 일반 버전).
//
// bwtrain.cpp 에서 파생. 마린 편 결과 재현을 위해 bwtrain 은 그대로 두고 따로 뺐다.
// 상수(체력·쿨다운·사거리)를 하드코딩하지 않고 게임 데이터에서 읽는다.
//
//   bwmicro --ally vulture --enemy zergling --allies 1 --enemies 6 --frame-skip 4
//
// 원본 주석:
// bwtrain.cpp — 강화학습용 스타크래프트 전투 환경.
//
// 파이썬(PyTorch)과 stdin/stdout 바이너리 프로토콜로 대화한다.
//   C++ → 파이썬 : 관측 + 직전 행동에 대한 보상 + 종료 여부
//   파이썬 → C++ : 마린별 행동 번호
//
// 게임을 8프레임 굴릴 때마다 한 번 주고받는다(사람 APM 수준).
// 에피소드가 끝나면 맵을 다시 안 읽고 유닛만 새로 스폰한다 — 이게 속도의 핵심.
//
//   bwtrain --data <mpq> --map <f.scx> [--marines 5] [--zealots 3]
//           [--episodes N] [--max-steps N] [--seed N] [--eval]
//           [--render-to <file.rgba>]     ← 지정 시 프레임을 파일로 저장(평가·촬영용)

#include "bwgame.h"
#include "ui/ui.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <unistd.h>

namespace bwgame { namespace ui {
void log_str(a_string s) { fwrite(s.data(), s.size(), 1, stderr); fflush(stderr); }
void fatal_error_str(a_string s) { fprintf(stderr, "fatal: %s\n", s.c_str()); std::exit(2); }
}}

using namespace bwgame;

// ── 유닛 상수는 게임 데이터에서 읽는다 (하드코딩하면 조합을 바꿀 때마다 틀린다)
static double ALLY_MAX_HP  = 40.0;
static double ENEMY_MAX_EFF= 160.0;
static int    ALLY_CD      = 15;      // 프레임
static int    ALLY_RANGE   = 128;     // 픽셀

// ── 관측·행동 규격 ──────────────────────────────────────────────────
static const int MAX_M = 12, MAX_Z = 16;
static const int N_ZOBS = 3, N_MOBS = 4;      // 관측에 넣는 적/아군 최대 수
static const int OBS_DIM = 5 + N_ZOBS * 6 + N_MOBS * 4;   // = 39 (적별 in_range 추가)
static const int N_ACTIONS = 12;
// 0 정지 / 1~8 8방향 이동 / 9 최근접 적 공격 / 10 최저HP 적 공격 / 11 후퇴

static const int DIRS[8][2] = {{0,-1},{1,-1},{1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1}};

struct tr_ui : ui_functions { using ui_functions::ui_functions; };

// ── 팀 하나에 필요한 전부. 양쪽이 각자 자기 상수·버퍼·집계를 갖는다.
struct Side {
    int owner = 0, n = 0;
    double max_hp = 1.0, foe_max_hp = 1.0;   // 내 유닛 / 상대 유닛 최대 HP
    int cd = 1, range = 1;                   // 내 무기 쿨다운 / 사거리
    std::vector<unit_t*>* me = nullptr;
    std::vector<unit_t*>* foe = nullptr;
    std::vector<float> obs;
    std::vector<uint8_t> alive, acts;
    std::vector<int> last_act;
    std::vector<unit_t*> last_tgt;
    long misfire = 0, misfire_prev = 0, act_count = 0;
    float reward = 0.f;
    void init(int o, int cnt, std::vector<unit_t*>* m, std::vector<unit_t*>* f) {
        owner = o; n = cnt; me = m; foe = f;
        obs.assign((size_t)n * OBS_DIM, 0.f);
        alive.assign(n, 0); acts.assign(n, 0);
        last_act.assign(n, -1); last_tgt.assign(n, nullptr);
    }
    void reset_ep() {
        std::fill(last_act.begin(), last_act.end(), -1);
        std::fill(last_tgt.begin(), last_tgt.end(), nullptr);
        misfire = misfire_prev = act_count = 0; reward = 0.f;
    }
};

// 쓸 만한 유닛만 이름으로 받는다. 오타로 엉뚱한 유닛이 나오는 것보다 낫다.
struct NamedUnit { const char* name; UnitTypes type; };
static const NamedUnit UNIT_TABLE[] = {
    { "marine",   UnitTypes::Terran_Marine },
    { "vulture",  UnitTypes::Terran_Vulture },
    { "firebat",  UnitTypes::Terran_Firebat },
    { "goliath",  UnitTypes::Terran_Goliath },
    { "zealot",   UnitTypes::Protoss_Zealot },
    { "dragoon",  UnitTypes::Protoss_Dragoon },
    { "corsair",  UnitTypes::Protoss_Corsair },
    { "zergling", UnitTypes::Zerg_Zergling },
    { "hydra",    UnitTypes::Zerg_Hydralisk },
    { "mutalisk", UnitTypes::Zerg_Mutalisk },
    { "scourge",  UnitTypes::Zerg_Scourge },
    { "lurker",   UnitTypes::Zerg_Lurker },
};
static bool lookup_unit(const char* n, UnitTypes& out) {
    for (auto& e : UNIT_TABLE) if (!strcmp(e.name, n)) { out = e.type; return true; }
    return false;
}

static int arg_int(int c, char** v, const char* n, int d) {
    for (int i = 1; i + 1 < c; ++i) if (!strcmp(v[i], n)) return atoi(v[i+1]);
    return d;
}
static const char* arg_str(int c, char** v, const char* n, const char* d) {
    for (int i = 1; i + 1 < c; ++i) if (!strcmp(v[i], n)) return v[i+1];
    return d;
}
static bool arg_flag(int c, char** v, const char* n) {
    for (int i = 1; i < c; ++i) if (!strcmp(v[i], n)) return true;
    return false;
}

static double hp_of(unit_t* u) {
    // ★ 보호막은 보호막이 있는 유닛(프로토스)만 더한다. 벌처·저글링도 shield_points 에
    //   쓰레기값 100 이 들어 있어서, 전엔 벌처 80→180, 저글링 35→135 로 읽혔다.
    //   관측의 체력 칸이 0~1 을 넘고, 처치·사망 보상이 2.3~3.9배로 부풀어 있었다 (9/22 수정).
    double h = (double)u->hp.raw_value / 256.0;
    if (u->unit_type->has_shield) h += (double)u->shield_points.raw_value / 256.0;
    return h;
}

static bool write_all(const void* p, size_t n) {
    const char* c = (const char*)p;
    while (n) { ssize_t w = write(1, c, n); if (w <= 0) return false; c += w; n -= w; }
    return true;
}
static bool read_all(void* p, size_t n) {
    char* c = (char*)p;
    while (n) { ssize_t r = read(0, c, n); if (r <= 0) return false; c += r; n -= r; }
    return true;
}

int main(int argc, char** argv) {
    const char* data_dir = arg_str(argc, argv, "--data", ".");
    const char* map_file = arg_str(argc, argv, "--map", nullptr);
    const char* render_to = arg_str(argc, argv, "--render-to", nullptr);
    const char* ally_s   = arg_str(argc, argv, "--ally",  "marine");
    const char* enemy_s  = arg_str(argc, argv, "--enemy", "zealot");
    const int n_marine   = std::min(MAX_M, arg_int(argc, argv, "--allies",
                              arg_int(argc, argv, "--marines", 5)));
    const int n_zealot   = std::min(MAX_Z, arg_int(argc, argv, "--enemies",
                              arg_int(argc, argv, "--zealots", 3)));
    // 맵 가운데는 지형이 막혀 있다. 뚫린 구역을 지정할 수 있게 뺀다.
    const int arg_cx     = arg_int(argc, argv, "--cx", -1);
    const int arg_cy     = arg_int(argc, argv, "--cy", -1);
    const int gap_lo     = arg_int(argc, argv, "--gap-lo", 220);
    const int gap_hi     = arg_int(argc, argv, "--gap-hi", 300);
    // 저글링(적) 시작 대형. 게임 규칙은 안 바꾸고 '어디에 세우나'만 바꾼다 (채점 케이스용).
    //   0 일렬(기본) · 1 위아래로 갈라짐 · 2 반원으로 감싸기 · 3 한 덩어리
    const int zform      = arg_int(argc, argv, "--zform", 0);
    // 럴커 판: 적은 버로우해서 고정, 아군에 디텍터(과학선)와 스팀을 붙인다
    const bool enemy_burrow = arg_flag(argc, argv, "--enemy-burrow");
    const bool use_detector = arg_flag(argc, argv, "--detector");
    const bool use_stim     = arg_flag(argc, argv, "--stim");
    // 적이 안 움직이는 판(럴커)에서는 '멀리 도망가서 시간만 보내기'가 가능하다.
    // 그러면 모든 판의 점수가 똑같아져서 학습 신호가 사라진다. 그래서 일정 스텝 동안
    // 아무도 교전 거리로 안 들어오면 그 자리에서 시간초과로 끝낸다.
    const int idle_steps    = arg_int(argc, argv, "--idle-steps", 0);
    const int engage_px     = arg_int(argc, argv, "--engage-px", 260);
    const int n_episodes = arg_int(argc, argv, "--episodes", 1000000);
    const int max_steps  = arg_int(argc, argv, "--max-steps", 120);   // 8프레임 × 120 = 960프레임 ≈ 40초
    const int frame_skip = arg_int(argc, argv, "--frame-skip", 8);
    const int base_seed  = arg_int(argc, argv, "--seed", 20260901);
    const bool eval_mode = arg_flag(argc, argv, "--eval");
    // ★ 셀프플레이: 플레이어1도 에이전트가 조종한다. 없으면 기존 동작 그대로.
    const bool selfplay  = arg_flag(argc, argv, "--selfplay");
    // ★ 사거리 판정을 엔진과 같은 방식(유닛 상자 간 거리)으로. 근접 유닛엔 필수.
    const bool box_range = arg_flag(argc, argv, "--box-range");
    const int W = arg_int(argc, argv, "--w", 640), H = arg_int(argc, argv, "--h", 480);

    // ── 보상 파라미터 (전부 명령줄로 조정 가능. 재빌드 없이 튜닝하려고) ──
    const float R_ENEMY_HP = arg_int(argc, argv, "--r-enemy-hp",  300) / 100.0f;  // 적 HP 1.0 깎을 때
    const float R_ALLY_HP  = arg_int(argc, argv, "--r-ally-hp",    30) / 100.0f;  // 아군 HP 1.0 잃을 때
    const float R_KILL     = arg_int(argc, argv, "--r-kill",      200) / 100.0f;  // 적 1기 처치
    const float R_DEATH    = arg_int(argc, argv, "--r-death",     100) / 100.0f;  // 아군 1기 사망
    const float R_WIN      = arg_int(argc, argv, "--r-win",      1000) / 100.0f;
    const float R_LOSE     = arg_int(argc, argv, "--r-lose",     1000) / 100.0f;
    const float R_TIMEOUT  = arg_int(argc, argv, "--r-timeout",  2500) / 100.0f;  // 도망 봉쇄의 핵심
    const float R_STEP     = arg_int(argc, argv, "--r-step",        1) / 1000.0f;
    const float R_MISFIRE  = arg_int(argc, argv, "--r-misfire",     2) / 100.0f;  // 사거리 밖 공격(헛방)

    if (!map_file) { fprintf(stderr, "usage: bwtrain --data <mpq> --map <f.scx> [...]\n"); return 1; }

    auto load = data_loading::data_files_directory(data_dir);
    game_player player(load);
    tr_ui ui(std::move(player));
    ui.create_window = false;
    ui.draw_ui_elements = false;
    ui.exit_on_close = false;
    ui.global_volume = 0;
    ui.load_data_file = [&](a_vector<uint8_t>& d, a_string fn) { load(d, std::move(fn)); };
    ui.init();

    state& st = ui.st;
    {
        data_loading::mpq_file<> ml(map_file);
        game_load_functions gl(st);
        gl.load_map([&](a_vector<uint8_t>& d, a_string fn) { ml(d, std::move(fn)); },
            [&]() {
                gl.setup_info.victory_condition = 0;   // 멜리 트리거 끄기(건물 없으면 즉사함)
                gl.setup_info.tournament_mode  = 1;    // 맵 자체 트리거도 끄기
                gl.setup_info.starting_units   = 0;
                gl.setup_info.create_no_units  = true;
                gl.setup_info.resource_type    = 1;
                gl.setup_info.starting_minerals = 0;
                for (int i = 0; i != 12; ++i) st.players[i].controller = player_t::controller_inactive;
                st.players[0].controller = player_t::controller_occupied;
                st.players[0].race = race_t::terran;
                st.players[1].controller = player_t::controller_occupied;
                st.players[1].race = race_t::protoss;
                st.lcg_rand_state = (uint32_t)base_seed;
            });
    }
    st.alliances[0][1] = 0; st.alliances[1][0] = 0;

    const int CX = arg_cx >= 0 ? arg_cx : (int)ui.game_st.map_width / 2;
    const int CY = arg_cy >= 0 ? arg_cy : (int)ui.game_st.map_height / 2;

    FILE* rf = nullptr;
    int cam_x = 0, cam_y = 0; bool cam_init = false;
    if (render_to) {
        rf = fopen(render_to, "wb");
        if (!rf) { fprintf(stderr, "render 파일 열기 실패: %s\n", render_to); return 1; }
        ui.resize(W, H);
        ui.set_image_data();
    }

    UnitTypes ally_id, enemy_id;
    if (!lookup_unit(ally_s, ally_id) || !lookup_unit(enemy_s, enemy_id)) {
        fprintf(stderr, "모르는 유닛 이름. 쓸 수 있는 것: ");
        for (auto& e : UNIT_TABLE) fprintf(stderr, "%s ", e.name);
        fprintf(stderr, "\n");
        return 1;
    }
    const unit_type_t* T_ALLY  = ui.get_unit_type(ally_id);
    const unit_type_t* T_ENEMY = ui.get_unit_type(enemy_id);
    {   // 상수는 전부 게임 데이터에서. 탱크·골리앗은 무기가 부포탑에 있다.
        const unit_type_t* wsrc = T_ALLY;
        if ((!T_ALLY->ground_weapon || T_ALLY->ground_weapon->damage_amount == 0)
            && T_ALLY->turret_unit_type) wsrc = T_ALLY->turret_unit_type;
        ALLY_MAX_HP   = (double)T_ALLY->hitpoints.raw_value / 256.0
                      + (T_ALLY->has_shield ? T_ALLY->shield_points : 0);
        ENEMY_MAX_EFF = (double)T_ENEMY->hitpoints.raw_value / 256.0
                      + (T_ENEMY->has_shield ? T_ENEMY->shield_points : 0);
        ALLY_CD    = wsrc->ground_weapon->cooldown;
        ALLY_RANGE = wsrc->ground_weapon->max_range;
        fprintf(stderr, "  아군 %s: HP %.0f · 쿨다운 %d · 사거리 %d\n",
                ally_s, ALLY_MAX_HP, ALLY_CD, ALLY_RANGE);
        fprintf(stderr, "  적   %s: HP %.0f\n", enemy_s, ENEMY_MAX_EFF);
    }

    Side S0, S1;
    {   // 상수는 팀별로 각자 게임 데이터에서 읽는다 (한쪽 기준 전역상수를 없앤다)
        auto consts_of = [&](const unit_type_t* t, double& hp, int& cd, int& rng) {
            const unit_type_t* w = t;
            if ((!t->ground_weapon || t->ground_weapon->damage_amount == 0)
                && t->turret_unit_type) w = t->turret_unit_type;
            hp  = (double)t->hitpoints.raw_value / 256.0 + (t->has_shield ? t->shield_points : 0);
            cd  = w->ground_weapon ? w->ground_weapon->cooldown  : 1;
            rng = w->ground_weapon ? w->ground_weapon->max_range : 1;
        };
        double hp; int cd, rng;
        consts_of(T_ALLY,  hp, cd, rng); S0.max_hp = hp; S0.cd = cd; S0.range = rng; S1.foe_max_hp = hp;
        consts_of(T_ENEMY, hp, cd, rng); S1.max_hp = hp; S1.cd = cd; S1.range = rng; S0.foe_max_hp = hp;
        if (selfplay) fprintf(stderr, "  [셀프플레이] P1 %s: HP %.0f · 쿨다운 %d · 사거리 %d\n",
                              enemy_s, S1.max_hp, S1.cd, S1.range);
    }

    const order_type_t* ORD_MOVE   = ui.get_order_type(Orders::Move);
    const order_type_t* ORD_ATTACK = ui.get_order_type(Orders::AttackUnit);
    const order_type_t* ORD_STOP   = ui.get_order_type(Orders::Stop);
    const order_type_t* ORD_BURROW = ui.get_order_type(Orders::Burrowing);
    const unit_type_t*  T_VESSEL   = ui.get_unit_type(UnitTypes::Terran_Science_Vessel);
    if (use_stim)     st.tech_researched[0][TechTypes::Stim_Packs]    = true;
    if (enemy_burrow) st.tech_researched[1][TechTypes::Lurker_Aspect] = true;
    unit_t* vessel = nullptr;

    std::vector<unit_t*> M(n_marine, nullptr), Z(n_zealot, nullptr);
    std::vector<double> prevM(n_marine, 0), prevZ(n_zealot, 0);
    S0.init(0, n_marine, &M, &Z);
    S1.init(1, n_zealot, &Z, &M);
    const bool dedup = !arg_flag(argc, argv, "--no-dedup");
    // 사거리 밖 공격을 무효화한다. 켜면 AI가 스스로 거리를 잡아야만 딜을 넣을 수 있다.
    const bool strict_fire = arg_flag(argc, argv, "--strict-fire");

    // 살아있는 유닛만 정리 (죽은 포인터는 st.player_units 에서 빠진다)
    auto compact = [&](std::vector<unit_t*>& v, int owner) {
        std::vector<unit_t*> cur;
        for (unit_t* u : ptr(st.player_units[owner])) if (u) cur.push_back(u);
        for (auto& p : v) {
            if (!p) continue;
            if (std::find(cur.begin(), cur.end(), p) == cur.end()) p = nullptr;
        }
    };

    auto clear_all = [&]() {
        for (int p = 0; p < 12; ++p) {
            std::vector<unit_t*> d;
            for (unit_t* u : ptr(st.player_units[p])) if (u) d.push_back(u);
            for (unit_t* u : d) ui.destroy_unit(u);
        }
    };

    // ── 대칭화된 3개 루프: 관측 / 행동 / 보상. 팀을 인자로 받는다. ──────────
    auto in_rng = [&](const Side& S, unit_t* u, unit_t* t, double cent_d) {
        if (!t) return false;
        if (box_range) return ui.units_distance(u, t) <= S.range;   // 엔진과 같은 상자거리
        return cent_d <= (double)S.range;                            // 기존(중심거리)
    };
    auto build_obs = [&](Side& S) {
        std::fill(S.obs.begin(), S.obs.end(), 0.f);
        for (int i = 0; i < S.n; ++i) {
            S.alive[i] = (*S.me)[i] ? 1 : 0;
            if (!(*S.me)[i]) continue;
            unit_t* u = (*S.me)[i];
            float* o = &S.obs[(size_t)i * OBS_DIM];
            o[0] = (float)(hp_of(u) / S.max_hp);
            o[1] = (float)(std::min(u->ground_weapon_cooldown, S.cd) / (double)S.cd);
            o[2] = 1.f;
            o[3] = (float)((u->position.x - CX) / 512.0);
            o[4] = (float)((u->position.y - CY) / 512.0);
            std::vector<std::pair<double, unit_t*>> zs;
            for (auto* z : *S.foe) if (z) {
                double dx = z->position.x - u->position.x, dy = z->position.y - u->position.y;
                zs.push_back({std::sqrt(dx*dx + dy*dy), z});
            }
            std::sort(zs.begin(), zs.end(), [](auto& a, auto& b){ return a.first < b.first; });
            for (int k = 0; k < N_ZOBS && k < (int)zs.size(); ++k) {
                unit_t* z = zs[k].second; float* q = o + 5 + k * 6;
                q[0] = (float)((z->position.x - u->position.x) / 256.0);
                q[1] = (float)((z->position.y - u->position.y) / 256.0);
                q[2] = (float)(zs[k].first / 256.0);
                q[3] = (float)(hp_of(z) / S.foe_max_hp);
                q[4] = 1.f;
                q[5] = in_rng(S, u, z, zs[k].first) ? 1.f : 0.f;
            }
            int c = 0;
            for (int j = 0; j < S.n && c < N_MOBS; ++j) {
                if (j == i || !(*S.me)[j]) continue;
                float* q = o + 5 + N_ZOBS * 6 + c * 4;
                q[0] = (float)(((*S.me)[j]->position.x - u->position.x) / 256.0);
                q[1] = (float)(((*S.me)[j]->position.y - u->position.y) / 256.0);
                q[2] = (float)(hp_of((*S.me)[j]) / S.max_hp);
                q[3] = 1.f; ++c;
            }
        }
    };
    auto apply_acts = [&](Side& S) {
        for (int i = 0; i < S.n; ++i) {
            unit_t* u = (*S.me)[i]; if (!u) continue;
            ++S.act_count;
            int a = S.acts[i] % N_ACTIONS;
            if (a == 0) {
                if (!dedup || S.last_act[i] != 0) { ui.set_unit_order(u, ORD_STOP); S.last_act[i] = 0; S.last_tgt[i] = nullptr; }
                continue;
            }
            if (a >= 1 && a <= 8) {
                int d = a - 1;
                ui.set_unit_order(u, ORD_MOVE,
                    xy(u->position.x + DIRS[d][0] * 96, u->position.y + DIRS[d][1] * 96));
                S.last_act[i] = a; S.last_tgt[i] = nullptr;
                continue;
            }
            unit_t* near_z = nullptr; double nd = 1e18;
            unit_t* weak_z = nullptr; double wh = 1e18;
            for (auto* z : *S.foe) if (z) {
                double dx = z->position.x - u->position.x, dy = z->position.y - u->position.y;
                double d2 = dx*dx + dy*dy;
                if (d2 < nd) { nd = d2; near_z = z; }
                double h = hp_of(z);
                if (h < wh) { wh = h; weak_z = z; }
            }
            auto ir = [&](unit_t* t) {
                if (!t) return false;
                double dx = t->position.x - u->position.x, dy = t->position.y - u->position.y;
                return in_rng(S, u, t, std::sqrt(dx*dx + dy*dy));
            };
            if (strict_fire && ((a == 9 && !ir(near_z)) || (a == 10 && !ir(weak_z)))) {
                ++S.misfire;
                if (!dedup || S.last_act[i] != 0) { ui.set_unit_order(u, ORD_STOP); S.last_act[i] = 0; S.last_tgt[i] = nullptr; }
                continue;
            }
            if (a == 9 && near_z) {
                if (!dedup || S.last_act[i] != 9 || S.last_tgt[i] != near_z) {
                    ui.set_unit_order(u, ORD_ATTACK, near_z);
                    S.last_act[i] = 9; S.last_tgt[i] = near_z;
                }
            }
            else if (a == 10 && weak_z) {
                if (!dedup || S.last_act[i] != 10 || S.last_tgt[i] != weak_z) {
                    ui.set_unit_order(u, ORD_ATTACK, weak_z);
                    S.last_act[i] = 10; S.last_tgt[i] = weak_z;
                }
            }
            else if (a == 11 && near_z) {
                S.last_act[i] = 11; S.last_tgt[i] = nullptr;
                double dx = u->position.x - near_z->position.x, dy = u->position.y - near_z->position.y;
                double L = std::max(1.0, std::sqrt(dx*dx + dy*dy));
                ui.set_unit_order(u, ORD_MOVE,
                    xy(u->position.x + (int)(dx / L * 96), u->position.y + (int)(dy / L * 96)));
            }
        }
    };
    // 보상은 한 번에 둘 다 계산한다 — prev 배열이 양쪽에서 공유되기 때문.
    auto compute_rewards = [&](int mf0, int mf1) {
        S0.reward = 0.f; S1.reward = 0.f;
        for (int i = 0; i < n_zealot; ++i) {
            double now = Z[i] ? hp_of(Z[i]) : 0.0;
            double d = prevZ[i] - now;
            if (d > 0) { S0.reward += (float)(d / S0.foe_max_hp) * R_ENEMY_HP;
                         S1.reward -= (float)(d / S1.max_hp)     * R_ALLY_HP; }
            if (prevZ[i] > 0 && now == 0.0) { S0.reward += R_KILL; S1.reward -= R_DEATH; }
            prevZ[i] = now;
        }
        for (int i = 0; i < n_marine; ++i) {
            double now = M[i] ? hp_of(M[i]) : 0.0;
            double d = prevM[i] - now;
            if (d > 0) { S0.reward -= (float)(d / S0.max_hp)     * R_ALLY_HP;
                         S1.reward += (float)(d / S1.foe_max_hp) * R_ENEMY_HP; }
            if (prevM[i] > 0 && now == 0.0) { S0.reward -= R_DEATH; S1.reward += R_KILL; }
            prevM[i] = now;
        }
        S0.reward -= R_STEP; S0.reward -= R_MISFIRE * mf0;
        S1.reward -= R_STEP; S1.reward -= R_MISFIRE * mf1;
    };
    auto emit_obs = [&](Side& S, char tag, int episode, int step, uint8_t done) {
        uint8_t t8 = (uint8_t)tag;
        if (!write_all(&t8, 1)) return false;
        int32_t e32 = episode, s32 = step;
        write_all(&e32, 4); write_all(&s32, 4);
        write_all(&S.reward, 4); write_all(&done, 1);
        uint8_t a8 = (uint8_t)S.n, b8 = (uint8_t)S.foe->size();
        write_all(&a8, 1); write_all(&b8, 1);
        write_all(S.obs.data(), S.obs.size() * sizeof(float));
        write_all(S.alive.data(), S.alive.size());
        return true;
    };
    auto read_acts = [&](Side& S, char want) {
        uint8_t atag = 0;
        if (!read_all(&atag, 1)) return false;
        if (atag != (uint8_t)want) {
            fprintf(stderr, "bwtrain: 프로토콜 오류 tag=%d (기대 %c)\n", atag, want);
            std::exit(4);
        }
        return read_all(S.acts.data(), S.acts.size());
    };

    int episode = 0;
    long total_steps = 0;
    fprintf(stderr, "bwtrain: ready. marines=%d zealots=%d obs=%d act=%d max_steps=%d skip=%d%s\n",
            n_marine, n_zealot, OBS_DIM, N_ACTIONS, max_steps, frame_skip, eval_mode ? " [EVAL]" : "");
    fprintf(stderr, "  제자리사격(strict-fire)=%s\n", strict_fire ? "ON" : "off");
    fprintf(stderr, "  보상: 적HP %.2f / 아군HP -%.2f / 처치 +%.1f / 사망 -%.1f / 승 +%.1f / 패 -%.1f / 시간초과 -%.1f / 스텝 -%.4f\n",
            R_ENEMY_HP, R_ALLY_HP, R_KILL, R_DEATH, R_WIN, R_LOSE, R_TIMEOUT, R_STEP);

    // 대형 k 번째 저글링의 자리. (sx, sy) 는 스폰 실패 때 교전 전체를 옮기는 양.
    auto zpos = [&](int i, int n, int gap, int spread, int sx, int sy) {
        const int vx = CX - gap / 2, vy = CY;               // 벌처 자리
        if (zform == 1) {                                   // 위아래 두 무리
            int half = (n + 1) / 2, g = i < half ? 0 : 1, k = i < half ? i : i - half;
            int m = g == 0 ? half : n - half;
            int oy = (g == 0 ? -1 : 1) * (gap * 6 / 10);
            return xy(CX + gap / 2 + sx, (int)(CY + oy + (k - (m - 1) / 2.0) * spread * 0.8) + sy);
        }
        if (zform == 2) {                                   // 벌처를 중심으로 한 반원 (오른쪽 -70°~+70°)
            double a = (n == 1 ? 0.0 : (-70.0 + 140.0 * i / (n - 1))) * 3.14159265 / 180.0;
            return xy((int)(vx + gap * std::cos(a)) + sx, (int)(vy + gap * std::sin(a)) + sy);
        }
        if (zform == 3) {                                   // 2열 × 3행 한 덩어리
            int c = i % 2, r = i / 2, rows = (n + 1) / 2;
            return xy(CX + gap / 2 + c * 26 + sx, (int)(CY + (r - (rows - 1) / 2.0) * 26) + sy);
        }
        return xy(CX + gap / 2 + sx, CY - (n - 1) * spread / 2 + i * spread + sy);
    };

    for (; episode < n_episodes; ++episode) {
        // ── 에피소드 리셋 ──
        clear_all();
        // 학습 시엔 배치를 흔들어 다양성을 준다. 평가 시엔 고정(공정한 비교).
        uint32_t s = (uint32_t)(base_seed + (eval_mode ? 0 : episode));
        st.lcg_rand_state = s;
        auto rnd = [&](int lo, int hi) {
            if (eval_mode) return (lo + hi) / 2;
            s = s * 22695477u + 1u;
            return lo + (int)(((s >> 16) & 0x7fff) % (uint32_t)(hi - lo + 1));
        };
        int gap = rnd(gap_lo, gap_hi), spread = rnd(34, 48);
        for (int i = 0; i < n_marine; ++i) {
            int y0 = CY - (n_marine - 1) * spread / 2;
            M[i] = ui.trigger_create_unit(T_ALLY,
                                          xy(CX - gap / 2, y0 + i * spread), 0);
        }
        for (int i = 0; i < n_zealot; ++i)
            Z[i] = ui.trigger_create_unit(T_ENEMY, zpos(i, n_zealot, gap, spread, 0, 0), 1);
        // 스폰 실패는 지형 때문에 가끔 생긴다. 위치를 조금씩 옮겨 재시도한다.
        // ★ 거리(gap)는 절대 줄이지 않는다 — 줄이면 접근 시간이 짧아져 판이 통째로 쉬워진다.
        //   대신 세로 간격을 좁히고, 그래도 안 되면 두 줄로 세운다.
        for (int retry = 0; retry < 16; ++retry) {
            bool ok = true;
            for (auto* u : M) if (!u) ok = false;
            for (auto* u : Z) if (!u) ok = false;
            if (ok) break;
            for (auto* u : M) if (u) ui.destroy_unit(u);
            for (auto* u : Z) if (u) ui.destroy_unit(u);
            std::fill(M.begin(), M.end(), nullptr);
            std::fill(Z.begin(), Z.end(), nullptr);
            int sp = std::max(20, spread - (retry % 8) * 3);
            int cols = 1 + (retry % 8) / 3;              // 3회마다 한 줄씩 늘린다
            // 8회까지 대형만 바꿔서 실패하면, 그 다음부터는 교전 전체를 통째로 옮긴다.
            // (거리는 그대로 유지한다 — 좁히면 판이 통째로 쉬워진다)
            int shx = retry < 8 ? 0 : (((retry - 8) % 3) - 1) * 56;
            int shy = retry < 8 ? 0 : (((retry - 8) / 3) - 1) * 72;
            int per  = (n_marine + cols - 1) / cols;
            for (int i = 0; i < n_marine; ++i) {
                int c = i / per, k = i % per;
                int y0 = CY - (per - 1) * sp / 2;
                M[i] = ui.trigger_create_unit(T_ALLY,
                          xy(CX - gap / 2 - c * 40 + shx, y0 + k * sp + shy), 0);
            }
            int zsp = std::max(28, spread - (retry % 8) * 3);
            int zcols = 1 + (retry % 8) / 4;
            int zper = (n_zealot + zcols - 1) / zcols;
            for (int i = 0; i < n_zealot; ++i) {
                if (zform != 0) {       // 대형 판은 모양을 바꾸면 다른 케이스가 된다 → 통째로만 옮긴다
                    Z[i] = ui.trigger_create_unit(T_ENEMY, zpos(i, n_zealot, gap, spread, shx, shy), 1);
                    continue;
                }
                int c = i / zper, k = i % zper;
                int y0 = CY - (zper - 1) * zsp / 2;
                Z[i] = ui.trigger_create_unit(T_ENEMY,
                          xy(CX + gap / 2 + c * 48 + shx, y0 + k * zsp + shy), 1);
            }
        }
        { bool ok = true;
          for (auto* u : M) if (!u) ok = false;
          for (auto* u : Z) if (!u) ok = false;
          if (!ok) { fprintf(stderr, "bwtrain: 스폰 16회 재시도 실패 (ep %d) — 이 에피소드 건너뜀\n", episode); continue; } }

        // 적은 학습 대상이 아니다. 고정 상대.
        const order_type_t* AM = ui.get_order_type(Orders::AttackMove);
        if (enemy_burrow) {
            // 럴커는 버로우해야 공격한다. 버로우하면 움직이지 않는다.
            for (auto* u : Z) ui.set_unit_order(u, ORD_BURROW);
            // 버로우가 끝날 때까지 몇 프레임 굴린다 (안 그러면 무기 없는 채로 시작한다)
            for (int f = 0; f < 40; ++f) ui.state_functions::next_frame();
            if (episode < 2) {
                int nb = 0; for (auto* u : Z) if (u && ui.u_burrowed(u)) ++nb;
                fprintf(stderr, "  [ep%d] 버로우 %d/%d\n", episode, nb, n_zealot);
            }
        } else if (!selfplay) {
            for (auto* u : Z) ui.set_unit_order(u, AM, xy(CX - gap / 2, CY));
        }
        (void)AM;
        // 디텍터: 버로우한 적을 보이게 하는 과학선. 전투에 참여하지 않는다
        // (럴커는 공중 공격이 없어서 안전하다). 관측·행동 대상도 아니다.
        if (use_detector) {
            vessel = ui.trigger_create_unit(T_VESSEL, xy(CX + gap / 2, CY), 0);
            if (vessel) ui.set_unit_order(vessel, ORD_MOVE, xy(CX + gap / 2, CY));
        }
        // 스팀은 자동으로 켠다. AI에게는 위치만 맡긴다 (손코딩 기준선도 같은 조건).
        int stim_at = -1000;

        S0.reset_ep(); S1.reset_ep();
        for (int i = 0; i < n_marine; ++i) prevM[i] = S0.max_hp;
        for (int i = 0; i < n_zealot; ++i) prevZ[i] = S1.max_hp;

        cam_init = false;
        int step = 0; uint8_t done = 0; int idle_run = 0;

        while (true) {
            compact(M, 0); compact(Z, 1);
            int nm = 0, nz = 0;
            for (auto* u : M) if (u) ++nm;
            for (auto* u : Z) if (u) ++nz;

            // 교전 거리 안에 아군이 하나라도 있나
            bool engaged = false;
            for (auto* u : M) {
                if (!u || engaged) continue;
                for (auto* z : Z) {
                    if (!z) continue;
                    double dx = z->position.x - u->position.x;
                    double dy = z->position.y - u->position.y;
                    if (dx * dx + dy * dy <= (double)engage_px * engage_px) { engaged = true; break; }
                }
            }
            if (engaged) idle_run = 0; else ++idle_run;

            if (nz == 0)            done = 1;   // 승리
            else if (nm == 0)       done = 2;   // 패배
            else if (step >= max_steps) done = 3;   // 시간 초과 = 패배 취급
            else if (idle_steps && idle_run >= idle_steps) done = 3;   // 안 붙으면 즉시 종료

            // ── 관측 만들기 (양쪽 대칭) ──
            build_obs(S0);
            if (selfplay) build_obs(S1);

            // ── 전송 ── 'O' = P0 관측, 'Q' = P1 관측(셀프플레이일 때만)
            if (!emit_obs(S0, 'O', episode, step, done)) return 0;
            if (selfplay && !emit_obs(S1, 'Q', episode, step, done)) return 0;

            if (done) break;

            // ── 행동 수신 ── 'A' = P0, 'B' = P1
            if (!read_acts(S0, 'A')) return 0;
            if (selfplay && !read_acts(S1, 'B')) return 0;

            // ── 행동 실행 (양쪽 대칭) ──
            apply_acts(S0);
            if (selfplay) apply_acts(S1);

            // ── 게임 진행 ──
            if (episode < 2 && step % 20 == 0) {
                double mhp = 0; int nm2 = 0;
                for (auto* u : M) if (u) { mhp += hp_of(u); ++nm2; }
                int lcd = -1, ldist = -1;
                for (auto* z : Z) if (z) {
                    lcd = z->ground_weapon_cooldown;
                    if (M[0]) {
                        double dx = z->position.x - M[0]->position.x;
                        double dy = z->position.y - M[0]->position.y;
                        ldist = (int)std::sqrt(dx*dx + dy*dy);
                    }
                    break;
                }
                fprintf(stderr, "   [ep%d s%3d] 마린 %d기 총HP %.0f · 럴커쿨다운 %d · 거리 %d\n",
                        episode, step, nm2, mhp, lcd, ldist);
            }
            if (use_stim && step - stim_at > 18) {   // 스팀 37프레임 → 주기적으로 다시
                stim_at = step;
                for (auto* u : M) {
                    if (!u || u->hp <= fp8::integer(12)) continue;
                    ui.unit_deal_damage(u, fp8::integer(10), nullptr, ~0);
                    u->stim_timer = 37;
                    ui.update_unit_speed(u);
                }
            }
            if (vessel) {   // 디텍터를 적 위에 붙여둔다
                long long vx = 0, vy = 0, vn = 0;
                for (auto* z : Z) if (z) { vx += z->position.x; vy += z->position.y; ++vn; }
                if (vn) ui.set_unit_order(vessel, ORD_MOVE,
                            xy((int)(vx / vn), (int)(vy / vn) - 40));
            }
            const int mf0 = (int)(S0.misfire - S0.misfire_prev); S0.misfire_prev = S0.misfire;
            const int mf1 = (int)(S1.misfire - S1.misfire_prev); S1.misfire_prev = S1.misfire;
            for (int f = 0; f < frame_skip; ++f) {
                ui.state_functions::next_frame();
                if (rf) {
                    // 카메라: 전 유닛 무게중심을 따라간다(고정 카메라면 후퇴 시 화면 밖으로 나간다).
                    long long sx = 0, sy = 0, cn = 0;
                    for (int pl = 0; pl < 2; ++pl)
                        for (unit_t* uu : ptr(st.player_units[pl])) {
                            if (!uu) continue; sx += uu->position.x; sy += uu->position.y; ++cn;
                        }
                    if (cn) {
                        int tx = (int)(sx / cn) - W / 2, ty = (int)(sy / cn) - H / 2;
                        // 지수 평활 — 카메라가 덜덜 떨리지 않게
                        cam_x = cam_init ? (cam_x * 7 + tx) / 8 : tx;
                        cam_y = cam_init ? (cam_y * 7 + ty) / 8 : ty;
                        cam_init = true;
                    }
                    ui.screen_pos = { cam_x, cam_y };
                    ui.update();
                    int pitch = 0, height = 0; uint32_t* px = nullptr;
                    std::tie(pitch, height, px) = ui.get_rgba_buffer();
                    for (int y = 0; y < H && y < height; ++y)
                        fwrite(px + (size_t)y * pitch, 4, W, rf);
                }
            }
            ++step; ++total_steps;

            // ── 보상 계산 (양쪽 한 번에 — prev 배열을 공유하므로) ──
            compact(M, 0); compact(Z, 1);
            compute_rewards(mf0, mf1);
        }

        // 종료 보상 — 'E' = P0(기존 그대로), 'F' = P1(셀프플레이일 때만 추가)
        float term = (done == 1) ? R_WIN : (done == 3 ? -R_TIMEOUT : -R_LOSE);
        uint8_t tag = 'E';
        write_all(&tag, 1); write_all(&term, 4); write_all(&done, 1);
        int nm = 0; for (auto* u : M) if (u) ++nm;
        uint8_t nm8 = (uint8_t)nm; write_all(&nm8, 1);   // 생존 마린 수(승률 이후의 주 지표)
        float mf = S0.act_count ? (float)S0.misfire / (float)S0.act_count : 0.f;
        write_all(&mf, 4);                               // 헛방 비율(진단)
        if (selfplay) {
            float term1 = (done == 2) ? R_WIN : (done == 3 ? -R_TIMEOUT : -R_LOSE);
            uint8_t tag1 = 'F';
            write_all(&tag1, 1); write_all(&term1, 4); write_all(&done, 1);
            int nz2 = 0; for (auto* u : Z) if (u) ++nz2;
            uint8_t nz8b = (uint8_t)nz2; write_all(&nz8b, 1);
            float mf1b = S1.act_count ? (float)S1.misfire / (float)S1.act_count : 0.f;
            write_all(&mf1b, 4);
        }
    }
    if (rf) fclose(rf);
    fprintf(stderr, "bwtrain: done. episodes=%d steps=%ld\n", episode, total_steps);
    return 0;
}
