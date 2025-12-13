"""
MANIKIN Training Logger
Forward/Backward 연산 추적, NaN 감지, 콘솔/TXT 로깅

Features:
- Forward pass: 각 모듈 출력의 shape, min, max, NaN 여부
- Backward pass: gradient 상태 (register_hook 사용)
- 콘솔 출력 + TXT 파일 저장
- Loss 항목별 추적 (Eq. 15)
"""

import os
import sys
import torch
import torch.nn as nn
from datetime import datetime
from collections import OrderedDict
import logging


class MANIKINLogger:
    """
    MANIKIN 학습 로거
    
    Usage:
        logger = MANIKINLogger(log_dir='Manikin/outputs/logs', name='train')
        logger.log_forward('NeuralNetwork', 'global_orient', tensor)
        logger.log_gradient('NeuralNetwork.fc_orient', grad_tensor)
        logger.log_loss(loss_dict, step=100)
        logger.save()
    """
    
    def __init__(self, log_dir='Manikin/outputs/logs', name='manikin', 
                 console_level='INFO', file_level='DEBUG'):
        """
        Args:
            log_dir: 로그 파일 저장 디렉토리
            name: 로거 이름 (파일명에 사용)
            console_level: 콘솔 출력 레벨 ('DEBUG', 'INFO', 'WARNING', 'ERROR')
            file_level: 파일 출력 레벨
        """
        self.log_dir = log_dir
        self.name = name
        os.makedirs(log_dir, exist_ok=True)
        
        # 타임스탬프
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'{name}_{timestamp}.txt')
        
        # Python logging 설정
        self.logger = logging.getLogger(f'MANIKIN_{name}')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, console_level))
        console_format = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # 파일 핸들러
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, file_level))
        file_format = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', 
                                        datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # 통계 저장
        self.forward_stats = OrderedDict()
        self.gradient_stats = OrderedDict()
        self.loss_history = []
        self.nan_events = []
        
        self.logger.info(f'MANIKIN Logger initialized. Log file: {self.log_file}')
    
    def _tensor_stats(self, tensor, name='tensor'):
        """텐서 통계 계산"""
        if tensor is None:
            return {'name': name, 'shape': None, 'nan': True, 'inf': True}
        
        with torch.no_grad():
            stats = {
                'name': name,
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype).replace('torch.', ''),
                'device': str(tensor.device),
                'min': tensor.min().item() if tensor.numel() > 0 else 0,
                'max': tensor.max().item() if tensor.numel() > 0 else 0,
                'mean': tensor.mean().item() if tensor.numel() > 0 else 0,
                'std': tensor.std().item() if tensor.numel() > 0 and tensor.numel() > 1 else 0,
                'nan': torch.isnan(tensor).any().item(),
                'inf': torch.isinf(tensor).any().item(),
                'nan_count': torch.isnan(tensor).sum().item(),
                'inf_count': torch.isinf(tensor).sum().item(),
            }
        return stats
    
    def _format_stats(self, stats, compact=False):
        """통계를 문자열로 포맷"""
        if stats['shape'] is None:
            return f"{stats['name']}: None"
        
        status = '✓'
        if stats['nan']:
            status = '✗NaN'
        elif stats['inf']:
            status = '✗Inf'
        
        if compact:
            return f"{stats['name']}: {stats['shape']} [{status}]"
        else:
            return (f"{stats['name']}: shape={stats['shape']}, "
                   f"range=[{stats['min']:.4f}, {stats['max']:.4f}], "
                   f"mean={stats['mean']:.4f}, std={stats['std']:.4f} [{status}]")
    
    # ========== Forward Pass Logging ==========
    
    def log_forward(self, module_name, tensor_name, tensor, level='DEBUG'):
        """Forward pass에서 텐서 로깅"""
        key = f'{module_name}.{tensor_name}'
        stats = self._tensor_stats(tensor, tensor_name)
        self.forward_stats[key] = stats
        
        msg = f'[FWD] {module_name} | {self._format_stats(stats)}'
        
        if stats['nan'] or stats['inf']:
            self.nan_events.append({
                'type': 'forward',
                'module': module_name,
                'tensor': tensor_name,
                'stats': stats
            })
            self.logger.warning(msg)
        else:
            getattr(self.logger, level.lower())(msg)
    
    def log_forward_batch(self, module_name, tensor_dict, level='DEBUG'):
        """여러 텐서를 한번에 로깅"""
        for name, tensor in tensor_dict.items():
            self.log_forward(module_name, name, tensor, level)
    
    # ========== Backward Pass Logging ==========
    
    def log_gradient(self, param_name, grad, level='DEBUG'):
        """Gradient 로깅"""
        stats = self._tensor_stats(grad, param_name)
        self.gradient_stats[param_name] = stats
        
        msg = f'[BWD] {self._format_stats(stats)}'
        
        if stats['nan'] or stats['inf']:
            self.nan_events.append({
                'type': 'gradient',
                'param': param_name,
                'stats': stats
            })
            self.logger.warning(msg)
        else:
            getattr(self.logger, level.lower())(msg)
    
    def register_gradient_hooks(self, model, module_names=None):
        """
        모델의 파라미터에 gradient hook 등록
        
        Args:
            model: nn.Module
            module_names: 추적할 모듈 이름 리스트 (None이면 전체)
        
        Returns:
            hooks: 해제용 hook 리스트
        """
        hooks = []
        
        for name, param in model.named_parameters():
            if module_names is not None:
                # 특정 모듈만 추적
                if not any(mn in name for mn in module_names):
                    continue
            
            if param.requires_grad:
                hook = param.register_hook(
                    lambda grad, n=name: self.log_gradient(n, grad)
                )
                hooks.append(hook)
        
        self.logger.info(f'Registered {len(hooks)} gradient hooks')
        return hooks
    
    def remove_hooks(self, hooks):
        """Hooks 해제"""
        for hook in hooks:
            hook.remove()
    
    # ========== Loss Logging ==========
    
    def log_loss(self, loss_dict, step, epoch=None, lr=None):
        """
        Loss 항목 로깅
        
        Args:
            loss_dict: {'total': x, 'L_ori': x, 'L_rot': x, ...}
            step: 현재 step
            epoch: 현재 epoch (optional)
            lr: learning rate (optional)
        """
        record = {'step': step, 'epoch': epoch, 'lr': lr, **loss_dict}
        self.loss_history.append(record)
        
        # 포맷팅
        header = f'[LOSS] step={step:,}'
        if epoch is not None:
            header += f', epoch={epoch}'
        if lr is not None:
            header += f', lr={lr:.2e}'
        
        loss_parts = []
        for k, v in loss_dict.items():
            if isinstance(v, (int, float)):
                loss_parts.append(f'{k}={v:.4f}')
        
        msg = f'{header} | {" | ".join(loss_parts)}'
        self.logger.info(msg)
    
    # ========== Summary & Reporting ==========
    
    def log_step_summary(self, step, include_forward=True, include_gradient=True):
        """Step 종료 시 요약 출력"""
        self.logger.info(f'\n{"="*70}')
        self.logger.info(f'STEP {step} SUMMARY')
        self.logger.info(f'{"="*70}')
        
        if include_forward and self.forward_stats:
            self.logger.info(f'\n[Forward Stats] ({len(self.forward_stats)} tensors)')
            nan_fwd = sum(1 for s in self.forward_stats.values() if s.get('nan'))
            inf_fwd = sum(1 for s in self.forward_stats.values() if s.get('inf'))
            self.logger.info(f'  NaN: {nan_fwd}, Inf: {inf_fwd}')
            
            # NaN/Inf 발생한 것만 상세 출력
            for key, stats in self.forward_stats.items():
                if stats.get('nan') or stats.get('inf'):
                    self.logger.warning(f'  ⚠ {key}: {self._format_stats(stats, compact=False)}')
        
        if include_gradient and self.gradient_stats:
            self.logger.info(f'\n[Gradient Stats] ({len(self.gradient_stats)} params)')
            nan_grad = sum(1 for s in self.gradient_stats.values() if s.get('nan'))
            inf_grad = sum(1 for s in self.gradient_stats.values() if s.get('inf'))
            self.logger.info(f'  NaN: {nan_grad}, Inf: {inf_grad}')
            
            # NaN/Inf 발생한 것만 상세 출력
            for key, stats in self.gradient_stats.items():
                if stats.get('nan') or stats.get('inf'):
                    self.logger.warning(f'  ⚠ {key}: {self._format_stats(stats, compact=False)}')
        
        self.logger.info(f'{"="*70}\n')
        
        # Clear for next step
        self.forward_stats.clear()
        self.gradient_stats.clear()
    
    def log_nan_report(self):
        """NaN 발생 이력 리포트"""
        if not self.nan_events:
            self.logger.info('No NaN/Inf events recorded.')
            return
        
        self.logger.warning(f'\n{"!"*70}')
        self.logger.warning(f'NaN/Inf REPORT: {len(self.nan_events)} events')
        self.logger.warning(f'{"!"*70}')
        
        for i, event in enumerate(self.nan_events):
            if event['type'] == 'forward':
                self.logger.warning(f'  [{i+1}] FWD {event["module"]}.{event["tensor"]}')
            else:
                self.logger.warning(f'  [{i+1}] BWD {event["param"]}')
    
    def info(self, msg):
        self.logger.info(msg)
    
    def warning(self, msg):
        self.logger.warning(msg)
    
    def debug(self, msg):
        self.logger.debug(msg)
    
    def error(self, msg):
        self.logger.error(msg)


class GradientTracker:
    """
    Backward gradient 추적을 위한 헬퍼 클래스
    특정 텐서의 gradient를 추적하고 로깅
    """
    
    def __init__(self, logger):
        self.logger = logger
        self.tracked_tensors = {}
        self.hooks = []
    
    def track(self, tensor, name):
        """
        텐서의 gradient 추적 등록
        
        Usage:
            tracker.track(output['elbow_pos'], 'elbow_pos')
        """
        if tensor.requires_grad:
            hook = tensor.register_hook(
                lambda grad, n=name: self.logger.log_gradient(f'tensor.{n}', grad)
            )
            self.hooks.append(hook)
            self.tracked_tensors[name] = tensor
    
    def clear(self):
        """모든 hook 해제"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.tracked_tensors.clear()


def create_logger(log_dir='Manikin/outputs/logs', name='train', 
                  console_level='INFO', file_level='DEBUG'):
    """
    Logger 생성 헬퍼 함수
    
    Args:
        log_dir: 로그 저장 디렉토리
        name: 로거 이름
        console_level: 콘솔 출력 레벨
        file_level: 파일 출력 레벨
    
    Returns:
        MANIKINLogger instance
    """
    return MANIKINLogger(
        log_dir=log_dir,
        name=name,
        console_level=console_level,
        file_level=file_level
    )
