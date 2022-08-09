#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <array>
#include <thread>
#include <future>

#include "Index.h"
#include "CopyableAtomic.h"

namespace neighborlists {

template<typename I, bool periodic>
struct CellAdjacency {
    static_assert(std::is_signed_v<typename I::value_type>, "Needs index to be signed!");

    bool withinBounds(const I &index, const typename I::GridDims &ijk) const {
        const bool nonNegative = std::all_of(begin(ijk), end(ijk), [](const auto &element) { return element >= 0; });
        bool nonExceeding = true;
        for(std::size_t dim = 0; dim < I::Dims; ++dim) {
            nonExceeding &= ijk[dim] < index[dim];
        }
        return nonNegative && nonExceeding;
    }

    void findEachAdjCell(const I &index, const typename I::GridDims &ijk, std::size_t dim, std::int32_t r,
                         std::vector<std::size_t> &adj) const {
        for (int ii = ijk[dim] - r; ii <= ijk[dim] + r; ++ii) {
            typename I::GridDims cix {ijk};
            cix[dim] = ii;
            if constexpr(periodic) {
                cix[dim] = (cix[dim] % index[dim] + index[dim]) % index[dim];  // wrap around
            }

            if(withinBounds(index, cix)) {
                if (dim > 0) {
                    findEachAdjCell(index, cix, dim - 1, r, adj);
                } else {
                    adj.push_back(index.index(cix));
                }
            }
        }
    }
    CellAdjacency() = default;
    CellAdjacency(const I &index, const std::int32_t radius) {
        typename I::GridDims nNeighbors {};
        for (std::size_t i = 0; i < I::Dims; ++i) {
            nNeighbors[i] = std::min(index[i], static_cast<decltype(index[i])>(2 * radius + 1));
        }
        // number of neighbors plus the cell itself
        const auto nAdjacentCells = 1 + std::accumulate(begin(nNeighbors), end(nNeighbors), 1, std::multiplies<>());
        // number of neighbors plus cell itself plus padding to save how many neighbors actually
        cellNeighbors = Index<2>{std::array<typename I::value_type, 2>{index.size(), static_cast<typename I::value_type>(1 + nAdjacentCells)}};
        cellNeighborsContent.resize(cellNeighbors.size());

        std::vector<std::size_t> adj;
        adj.reserve(1 + nAdjacentCells);
        for(std::size_t i = 0; i < index.size(); ++i) {
            adj.resize(0);
            const auto ijk = index.inverse(i);
            findEachAdjCell(index, ijk, I::Dims - 1, radius, adj);
            std::sort(adj.begin(), adj.end());
            adj.erase(std::unique(std::begin(adj), std::end(adj)), std::end(adj));

            const auto begin = cellNeighbors(i, 0);
            cellNeighborsContent[begin] = adj.size();
            std::copy(adj.begin(), adj.end(), &cellNeighborsContent.at(begin + 1));
        }
    }

    CellAdjacency(const CellAdjacency&) = delete;
    CellAdjacency &operator=(const CellAdjacency&) = delete;

    CellAdjacency(CellAdjacency&&) noexcept = default;
    CellAdjacency &operator=(CellAdjacency&&) noexcept = default;

    ~CellAdjacency() = default;

    [[nodiscard]] auto nNeighbors(auto cellIndex) const {
        return cellNeighborsContent[cellNeighbors(cellIndex, 0)];
    }

    [[nodiscard]] auto cellsBegin(auto cellIndex) const {
        return begin(cellNeighborsContent) + cellNeighbors(cellIndex, 1);
    }

    [[nodiscard]] auto cellsEnd(auto cellIndex) const {
        return cellsBegin(cellIndex) + nNeighbors(cellIndex);
    }

    // index of size (n_cells x (1 + nAdjacentCells)), where the first element tells how many adj cells are stored
    Index<2> cellNeighbors {};
    // backing vector of _cellNeighbors index of size (n_cells x (1 + nAdjacentCells))
    std::vector<std::size_t> cellNeighborsContent {};
};

template<int DIM, bool periodic, typename dtype>
class NeighborList {
public:
    using Index = neighborlists::Index<DIM, std::array<std::int32_t, DIM>>;
    NeighborList(std::array<dtype, DIM> gridSize, dtype interactionRadius, int nSubdivides = 2)
            : _gridSize(gridSize), nSubdivides(nSubdivides) {
        std::array<typename Index::value_type, DIM> nCells;
        for (int i = 0; i < DIM; ++i) {
            _cellSize[i] = interactionRadius / nSubdivides;
            if (gridSize[i] <= 0) {
                throw std::invalid_argument("grid sizes must be positive.");
            }
            nCells[i] = gridSize[i] / _cellSize[i];
        }
        _index = Index(nCells);
        head.resize(_index.size());
        _adjacency = CellAdjacency<Index, periodic>{_index, nSubdivides};
    }

    ~NeighborList() = default;
    NeighborList(const NeighborList&) = delete;
    NeighborList &operator=(const NeighborList&) = delete;

    NeighborList(NeighborList&&) noexcept = default;
    NeighborList &operator=(NeighborList&&) noexcept = default;

    template<std::random_access_iterator Iterator>
    void update(Iterator begin, Iterator end, std::uint32_t nJobs) {
        list.resize(std::distance(begin, end) + 1);
        std::fill(std::begin(list), std::end(list), 0);
        std::fill(std::begin(head), std::end(head), CopyableAtomic<std::size_t>());
        const auto updateOp = [this](std::size_t particleId, const auto &pos) {
            const auto boxId = positionToCellIndex(&pos.data[0]);

            // CAS
            auto &atomic = *head.at(boxId);
            auto currentHead = atomic.load();
            while (!atomic.compare_exchange_weak(currentHead, particleId + 1)) {}
            list[particleId + 1] = currentHead;
        };

        if (nJobs == 1) {
            std::size_t id = 0;
            for (std::random_access_iterator auto it = begin; it != end; ++it, ++id) {
                updateOp(id, *it);
            }
        } else {
            std::size_t grainSize = std::distance(begin, end) / nJobs;
            std::vector<std::jthread> jobs;
            for (int i = 0; i < nJobs - 1; ++i) {
                auto next = std::min(begin + grainSize * DIM, end);
                if (begin != next) {
                    std::size_t idStart = i * grainSize;
                    jobs.emplace_back([&updateOp, idStart, begin, next]() {
                        auto id = idStart;
                        for (std::random_access_iterator auto it = begin; it != next; ++it, ++id) {
                            updateOp(id, *it);
                        }
                    });
                }
                begin = next;
            }
            if (begin != end) {
                std::size_t idStart = (nJobs - 1) * grainSize;
                jobs.emplace_back([&updateOp, idStart, begin, end]() {
                    auto id = idStart;
                    for (std::random_access_iterator auto it = begin; it != end; ++it, ++id) {
                        updateOp(id, *it);
                    }
                });
            }
        }
    }

    template<typename Position>
    typename Index::GridDims gridPos(const Position &pos) const {
        typename Index::GridDims projections;
        for (auto i = 0U; i < DIM; ++i) {
            projections[i] = static_cast<typename Index::value_type>(
                    std::max(static_cast<dtype>(0.), static_cast<dtype>(std::floor((pos[i] + .5 * _gridSize[i]) / _cellSize[i]))));
            projections[i] = std::clamp(projections[i], static_cast<typename Index::value_type>(0),
                                        static_cast<typename Index::value_type>(_index[i] - 1));
        }
        return projections;
    }

    template<typename Position>
    std::uint32_t positionToCellIndex(const Position &pos) const {
        return _index.index(gridPos(pos));
    }

    auto nCellsTotal() const {
        return _index.size();
    }

    template<typename F>
    void forEachNeighborInCell(F &&func, const typename Index::GridDims &cellIndex) const {
        forEachNeighborInCell(std::forward<F>(func), _index(cellIndex));
    }

    template<bool all, typename F>
    void forEachNeighborInCell(F &&func, typename Index::value_type cellIndex) const {
        auto particleId = (*head.at(cellIndex)).load();
        while (particleId != 0) {
            const auto begin = _adjacency.cellsBegin(cellIndex);
            const auto end = _adjacency.cellsEnd(cellIndex);
            for (auto k = begin; k != end; ++k) {
                const auto neighborCellId = *k;
                auto neighborId = (*head.at(neighborCellId)).load();
                while (neighborId != 0) {
                    if constexpr(all) {
                        if(neighborId != particleId) {
                            func(particleId - 1, neighborId - 1);
                        }
                    } else {
                        if (neighborId > particleId) {
                            func(particleId - 1, neighborId - 1);
                        }
                    }
                    neighborId = list.at(neighborId);
                }
            }

            particleId = list.at(particleId);
        }
    }



    template<typename ParticleCollection, typename F>
    void forEachNeighbor(std::size_t particleId, ParticleCollection &collection, F &&fun) const {
        const auto &pos = collection.position(particleId);
        const auto gridPos = this->gridPos(&pos[0]);
        const auto gridId = _index.index(gridPos);
        const auto begin = _adjacency.cellsBegin(gridId);
        const auto end = _adjacency.cellsEnd(gridId);
        for (auto k = begin; k != end; ++k) {
            const auto neighborCellId = *k;

            auto neighborId = (*head.at(neighborCellId)).load();
            while (neighborId != 0) {
                if (neighborId - 1 != particleId) {
                    fun(neighborId - 1, collection.position(neighborId - 1), collection.typeOf(neighborId - 1),
                        collection.force(neighborId - 1));
                }
                neighborId = list.at(neighborId);
            }
        }
    }

    const std::array<dtype, DIM> &gridSize() const {
        return _gridSize;
    }

private:
    std::array<dtype, DIM> _cellSize{};
    std::array<dtype, DIM> _gridSize{};
    std::vector<CopyableAtomic<std::size_t>> head{};
    std::vector<std::size_t> list{};
    Index _index{};
    CellAdjacency<Index, periodic> _adjacency {};
    int nSubdivides;
    std::unordered_set<std::size_t> types {};
};

}

